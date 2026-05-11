# Franka Residual RL — Peg-in-Hole Assembly

基于 MuJoCo 的 Franka Emika Panda 机械臂销孔装配（peg-in-hole）残差强化学习。

## 演示

<video src="assets/peg_in_hole_demo.mp4" controls autoplay loop muted width="480"></video>

*SAC 训练 10000 步后，机械臂匀速插入销钉。step 523 成功，depth_error=2.9mm。*

## 概述

RL 策略（SAC）输出 1D 残差速度指令（垂直方向），叠加在基于 Jacobian 的 resolved-rate IK 控制器之上。控制器负责 XY 对齐和姿态锁定，RL 只控制插入深度。

```
观测 (6维): [z_error, v_z, Fz, Fx, Fy, last_action]
动作 (1维): v_z_cmd ∈ [-1, 1]  (正值=向下插入)
```

## 环境要求

```
Python >= 3.9
conda create -n franka_rl python=3.9
conda activate franka_rl
pip install -r requirements.txt
```

## 快速开始

### 环境冒烟测试

```bash
python scripts/test_rl_env.py
```

### 训练

```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Windows 下避免 OpenMP 冲突
python rl/train.py
```

### 评估

```bash
python rl/evaluate.py --model results/sac_peg_in_hole/<timestamp>/best_model/best_model.zip
```

### 可视化

```bash
python scripts/test_policy.py --model logs/final_model.zip
```

## 项目结构

```
├── assets/                  # MuJoCo 模型和场景 XML
│   └── franka_panda/
│       └── franka_emika_panda/
│           ├── panda.xml              # Franka Panda 机器人模型
│           └── scene_peg_in_hole.xml  # 销孔装配任务场景
├── configs/                 # YAML 配置文件
├── controllers/             # 底层控制器 (IK, admittance, Pinocchio)
├── envs/                    # Gymnasium 环境
│   └── peg_in_hole_env.py   # 核心 RL 环境
├── rl/                      # RL 训练和策略
│   ├── train.py             # SAC 训练入口
│   ├── evaluate.py          # 评估脚本
│   └── residual_policy.py   # 自定义特征提取器
└── scripts/                 # 调试和测试脚本
```

## 已知问题

- **Windows 多进程训练**: 需 `export KMP_DUPLICATE_LIB_OK=TRUE` 解决 OpenMP 库冲突
- **进度条**: 需 `pip install stable-baselines3[extra]` 或设置 `progress_bar=False`
