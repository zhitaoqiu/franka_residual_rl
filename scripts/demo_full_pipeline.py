"""Full pipeline demo: start from arbitrary pose, IK aligns XY, RL inserts.

Phase 1 — IK brings peg XY above hole center (Z held steady)
Phase 2 — RL policy takes over Z-axis insertion

Usage: python scripts/demo_full_pipeline.py
"""
import os, sys, time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
from envs.peg_in_hole_env import PegInHoleResidualEnv

# ====== Config ======
MODEL_PATH = "results/sac_peg_in_hole/20260512_092136/final_model.zip"
N_EPISODES = 5
DELAY = 0.03
# ====================

model_path = os.path.join(project_root, MODEL_PATH)

raw_env = PegInHoleResidualEnv(render_mode="human")
raw_env.disable_randomization()
model = SAC.load(model_path, env=raw_env)
wrapped_env = model.env  # Monitor + DummyVecEnv

with mujoco.viewer.launch_passive(raw_env.mj_model, raw_env.mj_data) as viewer:
    for ep in range(N_EPISODES):
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{N_EPISODES}")

        # -- Raw reset to random offset joint config --
        mujoco.mj_resetData(raw_env.mj_model, raw_env.mj_data)
        q0 = raw_env.aligned_q_nominal + raw_env.np_random.uniform(-0.25, 0.25, size=7)
        raw_env.mj_data.qpos[:7] = q0
        raw_env.mj_data.qvel[:7] = 0.0
        if raw_env.mj_model.nq > 8:
            raw_env.mj_data.qpos[7] = 0.04
            raw_env.mj_data.qpos[8] = 0.04
        raw_env.joint_target_q = q0.copy()
        raw_env.mj_data.ctrl[:7] = raw_env.joint_target_q
        if raw_env.mj_model.nu > 7:
            raw_env.mj_data.ctrl[7] = raw_env.gripper_hold_ctrl
        mujoco.mj_forward(raw_env.mj_model, raw_env.mj_data)

        # Init orientation & state
        _, tip_rot0 = raw_env._get_peg_tip_pose()
        raw_env.desired_tip_rot = raw_env._make_vertical_down_rot(tip_rot0)
        raw_env.ft_bias_world = np.zeros(6, dtype=np.float64)

        tip_pos, _ = raw_env._get_peg_tip_pose()
        raw_env.step_count = 0
        raw_env.last_action = 0.0
        raw_env.termination_reason = "running"
        raw_env.episode_log = []
        raw_env.goal_z = 0.015
        raw_env.start_z = float(tip_pos[2])
        raw_env.target_z = raw_env.start_z
        raw_env.target_xy = np.array([0.5, 0.0], dtype=np.float64)
        raw_env.prev_depth_error = raw_env._compute_depth_error(float(tip_pos[2]))

        # ====== Phase 1: IK alignment (XY + Z to nominal height) ======
        print("[Phase 1] IK aligning to hole center ...")
        nominal_z = 0.09  # same height the RL policy was trained from
        raw_env.target_z = nominal_z
        p1_steps = 0
        while p1_steps < 500:
            tip_pos, _ = raw_env._get_peg_tip_pose()
            xy_err = np.sqrt((tip_pos[0] - 0.5) ** 2 + tip_pos[1] ** 2) * 1000
            z_err = abs(tip_pos[2] - nominal_z) * 1000
            if xy_err < 3.0 and z_err < 5.0:
                break
            raw_env.apply_action(np.array([0.0], dtype=np.float32))
            p1_steps += 1
            viewer.sync()
            time.sleep(DELAY)

        obs = raw_env.get_observation()
        print(f"  aligned in {p1_steps} steps: XY={xy_err:.1f}mm, Z={z_err:.1f}mm")

        # ====== Phase 2: RL insertion ======
        print("[Phase 2] RL insertion ...")
        done = False
        p2_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = raw_env.step(action)
            p2_steps += 1
            viewer.sync()
            time.sleep(DELAY)
            done = terminated or truncated

        if info["termination_reason"] == "success":
            print(f"  SUCCESS | total={p1_steps + p2_steps} steps | "
                  f"force={info['contact_force']:.1f}N | "
                  f"depth_err={info['distance_to_target'] * 1000:.1f}mm")
        else:
            print(f"  FAIL ({info['termination_reason']}) | "
                  f"total={p1_steps + p2_steps} steps | "
                  f"force={info['contact_force']:.1f}N")

raw_env.close()
print("\n[Done]")
