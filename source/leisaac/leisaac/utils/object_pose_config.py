import os


def require_object_poses_path(object_poses_path: str | None, flow_name: str) -> str:
    """Validate the required object pose path for MVP flows."""
    if object_poses_path is None or not object_poses_path.strip():
        raise ValueError(
            f"{flow_name} requires --object_poses_path so the environment does not fall back to baked-in scene poses."
        )

    normalized_path = os.path.abspath(os.path.expanduser(object_poses_path.strip()))
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"{flow_name} object pose file does not exist: {normalized_path}")

    return normalized_path


def set_required_object_poses_path(env_cfg, object_poses_path: str | None, flow_name: str) -> str:
    """Validate and store the required object pose path on an env config."""
    normalized_path = require_object_poses_path(object_poses_path, flow_name)
    env_cfg.object_poses_path = normalized_path
    return normalized_path
