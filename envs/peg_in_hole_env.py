import numpy as np
from envs.mujoco_base import MujocoBase


class PegInHoleEnv(MujocoBase):
    def __init__(
        self,
        model_path: str,
        ee_body_name: str = "panda_hand",
        target_pos = None,
        success_threshold: float = 0.02,
        max_steps: int = 200,
    ):
        super().__init__(model_path=model_path, ee_body_name=ee_body_name)

        if target_pos is None:
            target_pos = np.array([0.50, 0.0, 0.20], dtype=np.float64)

        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        super().reset()
        return self._get_task_obs()

    def _get_task_obs(self):
        obs = self.get_obs()
        ee_pos = obs["ee_pos"]
        qpos = obs["qpos"]
        qvel = obs["qvel"]

        rel_pos = self.target_pos - ee_pos

        return {
            "qpos": qpos,
            "qvel": qvel,
            "ee_pos": ee_pos,
            "target_pos": self.target_pos.copy(),
            "rel_pos": rel_pos,
        }

    def compute_reward(self, obs):
        dist = np.linalg.norm(obs["rel_pos"])
        reward = -dist
        return reward

    def is_success(self, obs):
        dist = np.linalg.norm(obs["rel_pos"])
        return dist < self.success_threshold

    def step(self, ctrl, n_substeps: int = 5):
        self.step_count += 1
        self.step_sim(ctrl, n_substeps=n_substeps)

        obs = self._get_task_obs()
        reward = self.compute_reward(obs)
        success = self.is_success(obs)
        terminated = success
        truncated = self.step_count >= self.max_steps

        info = {
            "success": success,
            "step_count": self.step_count,
            "distance_to_target": float(np.linalg.norm(obs["rel_pos"]))
        }

        return obs, reward, terminated, truncated, info