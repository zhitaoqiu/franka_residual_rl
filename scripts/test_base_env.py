from envs.mujoco_base import MujocoBase
import numpy as np

MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"

env = MujocoBase(MODEL_PATH, ee_body_name="panda_hand")

obs = env.reset()
print("reset ok")
print("qpos shape:", obs["qpos"].shape)
print("qvel shape:", obs["qvel"].shape)
print("ee pos:", obs["ee_pos"])

ctrl = np.zeros(env.nu)
env.step_sim(ctrl, n_substeps=10)

obs = env.get_obs()
print("step ok")
print("ee pos after step:", obs["ee_pos"])