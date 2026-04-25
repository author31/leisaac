import importlib.util
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "leisaac" / "utils" / "object_pose_config.py"
MODULE_SPEC = importlib.util.spec_from_file_location("object_pose_config", MODULE_PATH)
object_pose_config = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(object_pose_config)


class RequireObjectPosesPathTest(unittest.TestCase):
    def test_missing_path_raises_clear_value_error(self):
        with self.assertRaisesRegex(ValueError, "requires --object_poses_path"):
            object_pose_config.require_object_poses_path(None, "Policy evaluation")

    def test_nonexistent_path_raises_file_not_found(self):
        with self.assertRaisesRegex(FileNotFoundError, "object pose file does not exist"):
            object_pose_config.require_object_poses_path("/tmp/does-not-exist.json", "State machine datagen")

    def test_set_required_object_poses_path_normalizes_and_stores_path(self):
        class EnvCfg:
            object_poses_path = None

        with tempfile.TemporaryDirectory() as tmpdir:
            pose_file = pathlib.Path(tmpdir) / "poses.json"
            pose_file.write_text("{}", encoding="utf-8")

            env_cfg = EnvCfg()
            resolved_path = object_pose_config.set_required_object_poses_path(
                env_cfg,
                str(pose_file),
                "State machine replay",
            )

        self.assertEqual(resolved_path, str(pose_file.resolve()))
        self.assertEqual(env_cfg.object_poses_path, str(pose_file.resolve()))


if __name__ == "__main__":
    unittest.main()
