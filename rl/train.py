# filepath: rl/train.py
"""
残差 RL 训练入口
使用 SAC 算法，适合连续动作空间的机器人任务
运行方式: python rl/train.py
"""
import os
import sys
import numpy as np
from datetime import datetime

# 把项目根目录加入 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from envs.peg_in_hole_env import PegInHoleResidualEnv
from rl.residual_policy import POLICY_KWARGS

# ============================================================
# 训练超参数配置
# ============================================================
CONFIG = dict(
    # --- 环境 ---
    n_envs=8,                    # 并行环境数量，AutoDL CPU 核心多可以加到 16
    max_episode_steps=300,       # 与 env 里的 max_steps 一致

    # --- SAC 核心超参 ---
    learning_rate=3e-4,
    buffer_size=500_000,         # 经验回放池大小
    learning_starts=10_000,      # 开始学习前先收集多少步
    batch_size=512,
    tau=0.005,                   # 软更新系数
    gamma=0.99,                  # 折扣因子
    train_freq=1,                # 每步采样后训练一次
    gradient_steps=1,
    ent_coef="auto",             # 自动调整熵系数，SAC 的核心优势

    # --- 训练规模 ---
    total_timesteps=2_000_000,   # 总训练步数，可以根据收敛情况调整

    # --- 日志与存档 ---
    log_dir="logs/sac_peg_in_hole",
    save_dir="results/sac_peg_in_hole",
    save_freq=50_000,            # 每隔多少步保存一次 checkpoint
    eval_freq=20_000,            # 每隔多少步评估一次
    n_eval_episodes=10,
)


def make_env(rank: int, seed: int = 0):
    """工厂函数：创建单个环境实例（给 SubprocVecEnv 用）"""
    def _init():
        env = PegInHoleResidualEnv(render_mode=None)  # 服务器上不渲染
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    # ============================================================
    # 1. 创建日志和存档目录
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, CONFIG["log_dir"], timestamp)
    save_dir = os.path.join(project_root, CONFIG["save_dir"], timestamp)
    best_model_dir = os.path.join(save_dir, "best_model")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print(f"[训练] 日志目录: {log_dir}")
    print(f"[训练] 模型目录: {save_dir}")

    # ============================================================
    # 2. 创建并行训练环境
    # ============================================================
    print(f"[训练] 创建 {CONFIG['n_envs']} 个并行环境...")
    train_env = SubprocVecEnv([
        make_env(rank=i, seed=42) for i in range(CONFIG["n_envs"])
    ])
    train_env = VecMonitor(train_env, log_dir)

    # 评估环境单独创建（单进程，方便 debug）
    eval_env = Monitor(PegInHoleResidualEnv(render_mode=None))

    # ============================================================
    # 3. 初始化 SAC 模型
    # ============================================================
    print("[训练] 初始化 SAC 模型...")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=CONFIG["learning_rate"],
        buffer_size=CONFIG["buffer_size"],
        learning_starts=CONFIG["learning_starts"],
        batch_size=CONFIG["batch_size"],
        tau=CONFIG["tau"],
        gamma=CONFIG["gamma"],
        train_freq=CONFIG["train_freq"],
        gradient_steps=CONFIG["gradient_steps"],
        ent_coef=CONFIG["ent_coef"],
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
        device="cuda",           # AutoDL 上用 GPU
    )

    print(f"[训练] 策略网络参数量: "
          f"{sum(p.numel() for p in model.policy.parameters()):,}")

    # ============================================================
    # 4. 配置回调
    # ============================================================
    # 定期评估并保存最优模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(CONFIG["eval_freq"] // CONFIG["n_envs"], 1),
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 定期保存 checkpoint（防止训练中断丢失进度）
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CONFIG["save_freq"] // CONFIG["n_envs"], 1),
        save_path=save_dir,
        name_prefix="sac_peg",
        save_replay_buffer=True,   # 保存经验池，断点续训用
        save_vecnormalize=False,
        verbose=1,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ============================================================
    # 5. 开始训练
    # ============================================================
    print(f"[训练] 开始训练，总步数: {CONFIG['total_timesteps']:,}")
    print("[训练] 用 tensorboard 监控: tensorboard --logdir " + log_dir)
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=callbacks,
            log_interval=100,        # 每 100 个 episode 打印一次日志
            reset_num_timesteps=True,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[训练] 手动中断，正在保存当前模型...")

    # ============================================================
    # 6. 保存最终模型
    # ============================================================
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    model.save_replay_buffer(os.path.join(save_dir, "final_replay_buffer"))
    print(f"[训练] 最终模型已保存至: {final_path}")

    train_env.close()
    eval_env.close()
    print("[训练] 完成！")


if __name__ == "__main__":
    main()