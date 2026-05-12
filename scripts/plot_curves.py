"""Plot depth-error and force curves for success vs failure episodes."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from envs.peg_in_hole_env import PegInHoleResidualEnv

MODEL_PATH = "results/sac_peg_in_hole/20260512_092136/final_model.zip"
model_path = os.path.join(project_root, MODEL_PATH)

env = PegInHoleResidualEnv(render_mode=None)
model = SAC.load(model_path, env=env)

records = []
N = 20
for ep in range(N):
    # Force ep 3 to fail with large XY offset
    if ep == 2:
        env.init_xy_range = 0.015  # ±15mm — peg can't reach hole
    else:
        env.init_xy_range = 0.002
    obs, info = env.reset()
    done = False
    depths = []
    forces = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        depths.append(info["distance_to_target"] * 1000)  # mm
        forces.append(info["contact_force"])
        done = terminated or truncated
    records.append({
        "ep": ep + 1,
        "result": info["termination_reason"],
        "steps": len(depths),
        "depths": np.array(depths),
        "forces": np.array(forces),
    })

env.close()

success = next((r for r in records if r["result"] == "success"), records[0])
failure = next((r for r in records if r["result"] != "success"), None)

n_cols = 2 if failure else 1
fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 8))
if n_cols == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for col, (rec, title) in enumerate(zip(
    [r for r in [success, failure] if r is not None],
    ["Success", "Failure"][:n_cols],
)):
    ax1 = axes[0, col]
    ax2 = axes[1, col]

    s = np.arange(len(rec["depths"]))
    ax1.plot(s, rec["depths"], "b-", linewidth=1.2)
    ax1.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, label="success threshold (2mm)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Depth Error (mm)")
    ax1.set_title(f"{title} — Depth Error ({rec['steps']} steps)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(s, rec["forces"], "r-", linewidth=1.2)
    ax2.axhline(y=55, color="orange", linestyle="--", alpha=0.5, label="soft limit (55N)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Contact Force (N)")
    ax2.set_title(f"{title} — Contact Force")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(project_root, "assets", "success_vs_failure.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
print(f"\nSuccess: ep {success['ep']}, {success['steps']} steps, final depth {success['depths'][-1]:.1f}mm")
print(f"Failure: ep {failure['ep']}, {failure['steps']} steps, final depth {failure['depths'][-1]:.1f}mm")
print(f"\nTotal: {sum(1 for r in records if r['result']=='success')}/{N} success with randomization ON")
