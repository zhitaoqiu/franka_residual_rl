"""Visualize a trained SAC policy on the peg-in-hole task.

Usage: python scripts/enjoy.py
"""
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from envs.peg_in_hole_env import PegInHoleResidualEnv

# ====== 改这里 ======
MODEL_PATH = "results/sac_peg_in_hole/20260512_092136/final_model.zip"
N_EPISODES = 5
DELAY = 0.04  # 每步间隔秒数，越小越快
# ====================

model_path = os.path.join(project_root, MODEL_PATH)

if not os.path.exists(model_path):
    print(f"ERROR: model not found at {model_path}")
    sys.exit(1)

print(f"[Enjoy] model: {model_path}")
print(f"[Enjoy] episodes: {N_EPISODES}")

env = PegInHoleResidualEnv(render_mode="human")
model = SAC.load(model_path, env=env)

with mujoco.viewer.launch_passive(env.unwrapped.mj_model, env.unwrapped.mj_data) as viewer:
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        step = 0

        print(f"\n--- Episode {ep + 1}/{N_EPISODES} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            viewer.sync()
            time.sleep(DELAY)
            done = terminated or truncated

        if info["termination_reason"] == "success":
            print(f"  SUCCESS | steps={step} | force={info['contact_force']:.1f}N | "
                  f"depth_err={info['distance_to_target']*1000:.1f}mm")
        else:
            print(f"  FAIL ({info['termination_reason']}) | steps={step} | "
                  f"force={info['contact_force']:.1f}N")

env.close()
print("\n[Enjoy] done!")