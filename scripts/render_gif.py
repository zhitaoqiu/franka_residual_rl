"""Render policy rollout frames and save as animated GIF."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
from PIL import Image
import mujoco
from stable_baselines3 import SAC

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from envs.peg_in_hole_env import PegInHoleResidualEnv

MODEL_PATH = os.path.join(project_root, "results/sac_peg_in_hole/20260512_092136/final_model.zip")
OUTPUT = os.path.join(project_root, "assets/peg_in_hole_demo.gif")

# Check model exists
if not os.path.exists(MODEL_PATH):
    # Try to find any model
    results_dir = os.path.join(project_root, "results/sac_peg_in_hole")
    if os.path.isdir(results_dir):
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                if f.endswith(".zip"):
                    MODEL_PATH = os.path.join(root, f)
                    break
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No model found. Expected at {MODEL_PATH}")
        sys.exit(1)

print(f"Model: {MODEL_PATH}")

# Create env and load model
env = PegInHoleResidualEnv(render_mode=None)
env.disable_randomization()
model = SAC.load(MODEL_PATH, env=env)

# Setup offscreen renderer
renderer = mujoco.Renderer(env.mj_model, height=480, width=600)

# Collect frames
frames = []
obs, info = env.reset()
done = False
step = 0

print("Rendering frames...")
while not done and step < 600:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Render
    renderer.update_scene(env.mj_data)
    img = renderer.render()
    frames.append(Image.fromarray(img))
    step += 1

    if step % 50 == 0:
        print(f"  frame {step}...")

renderer.close()
env.close()

print(f"Collected {len(frames)} frames.")
print(f"Episode result: {info.get('termination_reason', 'unknown')}, "
      f"depth_err={info.get('distance_to_target', 0)*1000:.1f}mm")

# Downsample to keep GIF size reasonable (target ~60 frames)
if len(frames) > 80:
    skip = len(frames) // 60
    frames = frames[::skip]
    print(f"Downsampled to {len(frames)} frames")

# Save GIF
print(f"Saving GIF to {OUTPUT}...")
frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=frames[1:],
    duration=50,  # 50ms per frame = 20fps
    loop=0,
    optimize=True,
)
file_size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Done! {len(frames)} frames, {file_size_mb:.1f} MB")
