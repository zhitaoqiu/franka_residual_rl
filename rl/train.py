"""
SAC training entry point for peg-in-hole residual RL.
Usage: python rl/train.py
"""
import os
import sys
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from envs.peg_in_hole_env import PegInHoleResidualEnv
from rl.residual_policy import POLICY_KWARGS

CONFIG = dict(
    n_envs=4,
    max_episode_steps=600,

    learning_rate=3e-4,
    buffer_size=200_000,
    learning_starts=5_000,
    batch_size=512,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",

    total_timesteps=200_000,
    log_dir="logs/sac_peg_in_hole",
    save_dir="results/sac_peg_in_hole",
    save_freq=40_000,
    eval_freq=20_000,
    n_eval_episodes=10,
)


def make_env(rank: int = 0, seed: int = 42, render: bool = False, eval_mode: bool = False):
    def _init():
        env = PegInHoleResidualEnv(render_mode="human" if render else None)
        if eval_mode:
            env.disable_randomization()
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, CONFIG["log_dir"], timestamp)
    save_dir = os.path.join(project_root, CONFIG["save_dir"], timestamp)
    best_model_dir = os.path.join(save_dir, "best_model")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Train] log_dir: {log_dir}")
    print(f"[Train] save_dir: {save_dir}")

    # Training envs
    print(f"[Train] creating {CONFIG['n_envs']} environments...")
    train_env = DummyVecEnv([make_env(rank=i) for i in range(CONFIG["n_envs"])])
    train_env = VecMonitor(train_env, log_dir)

    # Eval env (single, separate, deterministic for clean progress tracking)
    eval_env = DummyVecEnv([make_env(rank=999, seed=999, eval_mode=True)])
    eval_env = VecMonitor(eval_env)

    print("[Train] initializing SAC...")
    model = SAC(
        "MlpPolicy",
        train_env,
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
        device="cpu",
    )

    print(f"[Train] policy params: {sum(p.numel() for p in model.policy.parameters()):,}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(CONFIG["eval_freq"] // CONFIG["n_envs"], 1),
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CONFIG["save_freq"] // CONFIG["n_envs"], 1),
        save_path=save_dir,
        name_prefix="sac_peg",
        save_replay_buffer=True,
        verbose=1,
    )

    print(f"[Train] starting {CONFIG['total_timesteps']:,} timesteps...")
    print(f"[Train] tensorboard --logdir {log_dir}")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=CallbackList([eval_callback, checkpoint_callback]),
            log_interval=10,
            reset_num_timesteps=True,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[Train] interrupted, saving...")

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    model.save_replay_buffer(os.path.join(save_dir, "final_replay_buffer"))
    print(f"[Train] final model saved to {final_path}")

    train_env.close()
    eval_env.close()
    print("[Train] done!")


if __name__ == "__main__":
    main()
