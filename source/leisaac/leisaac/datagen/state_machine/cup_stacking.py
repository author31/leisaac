"""State machine for the Franka cup-stacking task."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_BLUE_CUP_NAME = "blue_cup"
_PINK_CUP_NAME = "pink_cup"
_EE_BODY_NAME = "panda_hand"

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

_MAX_CARTESIAN_DELTA = 0.018
_MAX_ROT_DELTA = 0.08

_HOVER_Z_OFFSET = 0.15
_GRASP_Z_OFFSET = 0.08
_LIFT_Z_OFFSET = 0.2
_GRIPPER_DOWN_ROLL_W = math.pi
_GRIPPER_DOWN_PITCH_W = 0.0

_SUCCESS_X_RANGE = (-0.05, 0.05)
_SUCCESS_Y_RANGE = (-0.05, 0.05)

_FRANKA_REST_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 4.0,
    "panda_joint3": 0.0,
    "panda_joint4": -3.0 * math.pi / 4.0,
    "panda_joint5": 0.0,
    "panda_joint6": math.pi / 2.0,
    "panda_joint7": math.pi / 4.0,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}


def _constant_gripper(num_envs: int, device: torch.device, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), value, device=device)


def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale


def _shortest_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[:, 0:1] < 0.0, -quat, quat)


def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) > 0:
            return int(body_ids[0])

    body_names = getattr(robot.data, "body_names", None)
    if body_names is not None and body_name in body_names:
        return body_names.index(body_name)

    return -1

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class CupStackingStateMachine(StateMachineBase):
    """Scripted Franka policy for placing the blue cup on the pink cup.

    The cup-stacking environment inherits ``SingleArmFrankaTaskEnvCfg``.  When
    configured with the Franka keyboard/gamepad action setup, the action vector
    is:

    ``[dx, dy, dz, drot_x, drot_y, drot_z, d_panda_joint1, gripper]``.

    This state machine therefore tracks world-space waypoints internally and
    emits clipped relative position and rotation-vector deltas in the robot-root
    frame.
    """

    MAX_STEPS: int = 720

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._ee_body_idx: int = -1
        self._initial_ee_pos_w: torch.Tensor | None = None
        self._rest_ee_pos_w: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._home_start_pos: torch.Tensor | None = None
        self._blue_lift_target_w: torch.Tensor | None = None
        self._pink_above_target_w: torch.Tensor | None = None
        self._pink_stack_target_w: torch.Tensor | None = None
        self._pink_retreat_target_w: torch.Tensor | None = None
        self._gripper_down_yaw_w: torch.Tensor | None = None
        self._event: int = 0
        self._events_dt = [
            160,  # Phase 0: Move above the blue cup
            80,  # Phase 1: Approach down to the blue cup
            20,  # Phase 2: Close gripper to grasp
            100,  # Phase 3: Lift blue cup upward
            80,  # Phase 4: Move blue cup above the pink cup
            30,  # Phase 5: Lower/place and release
            80,  # Phase 6: Move up and away
        ]

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """Record the Franka rest-pose joint state and end-effector position."""
        robot = env.scene["robot"]
        self._ee_body_idx = _find_body_index(robot, _EE_BODY_NAME)
        joint_names = list(robot.data.joint_names)

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _FRANKA_REST_JOINT_POS:
                self._rest_joint_pos[:, idx] = _FRANKA_REST_JOINT_POS[name]

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_w = self._ee_pos_w(robot).clone()

    def check_success(self, env) -> bool:
        """Return True when the blue cup is centered over and above the pink cup."""
        blue_cup_pos = env.scene[_BLUE_CUP_NAME].data.root_pos_w - env.scene.env_origins
        pink_cup_pos = env.scene[_PINK_CUP_NAME].data.root_pos_w - env.scene.env_origins

        done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        done = torch.logical_and(done, blue_cup_pos[:, 0] < pink_cup_pos[:, 0] + _SUCCESS_X_RANGE[1])
        done = torch.logical_and(done, blue_cup_pos[:, 0] > pink_cup_pos[:, 0] + _SUCCESS_X_RANGE[0])
        done = torch.logical_and(done, blue_cup_pos[:, 1] < pink_cup_pos[:, 1] + _SUCCESS_Y_RANGE[1])
        done = torch.logical_and(done, blue_cup_pos[:, 1] > pink_cup_pos[:, 1] + _SUCCESS_Y_RANGE[0])
        return bool(done.all().item())

    def pre_step(self, env) -> None:
        """No direct joint override is needed for the cup-stacking phases."""

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        blue_cup_pos_w = env.scene[_BLUE_CUP_NAME].data.root_pos_w.clone()
        pink_cup_pos_w = env.scene[_PINK_CUP_NAME].data.root_pos_w.clone()

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        if step == 0:
            if self._event == 0:
                self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        target_quat_w = self._gripper_down_quat_w(robot, num_envs, device, robot.data.root_quat_w.dtype)
        if self._event == 0:
            target_pos_w, gripper_cmd = self._phase_move_above_target(blue_cup_pos_w, num_envs, device)
        elif self._event == 1:
            target_pos_w, gripper_cmd = self._phase_approach_blue(blue_cup_pos_w, num_envs, device)
        elif self._event == 2:
            target_pos_w, gripper_cmd = self._phase_grasp_blue(blue_cup_pos_w, num_envs, device)
        elif self._event == 3:
            target_pos_w, gripper_cmd = self._phase_lift_blue(blue_cup_pos_w, num_envs, device)
        elif self._event == 4:
            target_pos_w, gripper_cmd = self._phase_move_above_pink(pink_cup_pos_w, num_envs, device)
        elif self._event == 5:
            target_pos_w, gripper_cmd = self._phase_lower_to_release(pink_cup_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_lift_away(pink_cup_pos_w, num_envs, device)

        return self._relative_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    def _phase_move_above_target(self, blue_cup_pos_w, num_envs, device):
        target_pos_w = blue_cup_pos_w.clone()
        target_pos_w[:, 2] += _HOVER_Z_OFFSET
        if self._initial_ee_pos_w is not None:
            denom = max(self._events_dt[0] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target_pos_w
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_approach_blue(self, blue_cup_pos_w, num_envs, device):
        target_pos_w = blue_cup_pos_w.clone()
        target_pos_w[:, 2] += _GRASP_Z_OFFSET
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_grasp_blue(self, blue_cup_pos_w, num_envs, device):
        target_pos_w = blue_cup_pos_w.clone()
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift_blue(self, blue_cup_pos_w, num_envs, device):
        target_pos_w = blue_cup_pos_w.clone()
        target_pos_w[:, 2] += _LIFT_Z_OFFSET
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_move_above_pink(self, pink_cup_pos_w, num_envs, device):
        target_pos_w = pink_cup_pos_w.clone()
        target_pos_w[:, 2] += _LIFT_Z_OFFSET
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lower_to_release(self, pink_cup_pos_w, num_envs, device):
        target_pos_w = pink_cup_pos_w.clone()
        target_pos_w[:, 2] += _LIFT_Z_OFFSET
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift_away(self, pink_cup_pos_w, num_envs, device):
        target_pos_w = pink_cup_pos_w.clone()
        target_pos_w[:, 2] += _LIFT_Z_OFFSET
        return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def advance(self) -> None:
        """Advance the scripted timeline."""
        if self._episode_done:
            return

        self._step_count += 1
        if self._step_count < self._events_dt[self._event]:
            return

        self._event += 1
        self._step_count = 0
        if self._event >= len(self._events_dt):
            self._episode_done = True

    def reset(self) -> None:
        """Reset per-episode state while keeping setup-time Franka calibration."""
        self._step_count = 0
        self._episode_done = False
        self._event = 0
        self._initial_ee_pos_w = None
        self._home_start_pos = None
        self._blue_lift_target_w = None
        self._pink_above_target_w = None
        self._pink_stack_target_w = None
        self._pink_retreat_target_w = None
        self._gripper_down_yaw_w = None

    # ------------------------------------------------------------------
    def _ee_pos_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_pos_w[:, body_idx, :]

    def _ee_quat_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_quat_w[:, body_idx, :]

    def _relative_franka_action(
        self,
        env,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        gripper_cmd: torch.Tensor,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        root_quat_inv = quat_inv(root_quat_w)

        target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
        ee_pos_root = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
        delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)

        delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
        delta_rot_w = axis_angle_from_quat(delta_quat_w)
        delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)

        base_joint_delta = torch.zeros(env.num_envs, 1, device=env.device, dtype=delta_pos_root.dtype)
        return torch.cat([delta_pos_root, delta_rot_root, base_joint_delta, gripper_cmd], dim=-1)

    def _gripper_down_quat_w(
        self, robot, num_envs: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._gripper_down_yaw_w is None or self._gripper_down_yaw_w.shape[0] != num_envs:
            self._gripper_down_yaw_w = self._current_hand_heading_yaw_w(robot).clone()

        roll = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W, device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        yaw = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    def _current_hand_heading_yaw_w(self, robot) -> torch.Tensor:
        quat_w = self._ee_quat_w(robot)
        local_x = torch.zeros(quat_w.shape[0], 3, device=quat_w.device, dtype=quat_w.dtype)
        local_y = torch.zeros_like(local_x)
        local_x[:, 0] = 1.0
        local_y[:, 1] = 1.0

        hand_x_w = quat_apply(quat_w, local_x)
        hand_y_w = quat_apply(quat_w, local_y)
        yaw_from_x = torch.atan2(hand_x_w[:, 1], hand_x_w[:, 0])
        yaw_from_y = torch.atan2(hand_y_w[:, 0], -hand_y_w[:, 1])

        x_is_horizontal = torch.linalg.norm(hand_x_w[:, :2], dim=-1) > 1e-4
        return torch.where(x_is_horizontal, yaw_from_x, yaw_from_y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count
