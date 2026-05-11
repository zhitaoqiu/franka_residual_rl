"""Test the PegInHoleResidualEnv with a basic step loop."""
import numpy as np
from envs.peg_in_hole_env import PegInHoleResidualEnv

env = PegInHoleResidualEnv(render_mode=None)

obs, info = env.reset()
print("reset ok")
print("target_xy:", info["target_xy"])
print("goal_z:", info["goal_z"])
print("obs:", obs)

for i in range(10):
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={i}, reward={reward:.4f}, z_tip={info['z_tip']:.4f}, "
          f"depth_error={info['depth_error']:.4f}, fz={info['fz']:.1f}")

    if terminated or truncated:
        print("episode end:", info["termination_reason"])
        break

env.close()
