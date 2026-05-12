# filepath: rl/evaluate.py
"""
训练结果评估脚本
用法: python rl/evaluate.py --model results/sac_peg_in_hole/xxx/best_model/best_model.zip
"""
import os
import sys
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from stable_baselines3 import SAC
from envs.peg_in_hole_env import PegInHoleResidualEnv


def evaluate(model_path: str, n_episodes: int = 20, render: bool = False):
    render_mode = "human" if render else None
    env = PegInHoleResidualEnv(render_mode=render_mode)
    env.disable_randomization()

    print(f"[评估] 加载模型: {model_path}")
    model = SAC.load(model_path, env=env, device="cuda")

    success_count = 0
    rewards = []
    forces = []
    distances = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_forces = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_forces.append(info["contact_force"])
            done = terminated or truncated

        rewards.append(ep_reward)
        forces.append(np.mean(ep_forces))
        distances.append(info["distance_to_target"])

        if info["termination_reason"] == "success":
            success_count += 1
            status = "成功"
        else:
            status = f"失败 ({info['termination_reason']})"

        print(f"  Episode {ep+1:02d}: {status} | "
              f"奖励={ep_reward:.2f} | "
              f"平均力={np.mean(ep_forces):.1f}N | "
              f"最终距离={info['distance_to_target']*1000:.1f}mm")

    print("\n" + "="*50)
    print(f"[结果] 成功率: {success_count}/{n_episodes} = {success_count/n_episodes*100:.1f}%")
    print(f"[结果] 平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"[结果] 平均接触力: {np.mean(forces):.1f} N")
    print(f"[结果] 平均最终距离: {np.mean(distances)*1000:.1f} mm")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路径 (.zip)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true", help="是否渲染（本地用）")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render)