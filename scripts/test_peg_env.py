import numpy as np
from envs.peg_in_hole_env import PegInHoleEnv

MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"

env = PegInHoleEnv(
    model_path=MODEL_PATH,
    ee_body_name="panda_hand",
    target_pos=[0.50, 0.0, 0.20],
    success_threshold=0.02,
    max_steps=50,
)

obs = env.reset()
print("reset ok")
print("ee_pos:", obs["ee_pos"])
print("target_pos:", obs["target_pos"])
print("rel_pos:", obs["rel_pos"])

for i in range(10):
    ctrl = np.zeros(env.nu)
    obs, reward, terminated, truncated, info = env.step(ctrl, n_substeps=5)
    print(f"step={i}, reward={reward:.4f}, dist={info['distance_to_target']:.4f}")

    if terminated or truncated:
        print("episode end:", info)
        break