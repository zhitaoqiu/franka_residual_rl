# filepath: rl/residual_policy.py
"""
残差策略网络定义
使用 stable-baselines3 的 SAC，自定义网络结构
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym


class ResidualFeaturesExtractor(BaseFeaturesExtractor):
    """
    自定义特征提取器
    输入 obs: [z_error, v_z, Fz, Fx, Fy, last_action] (6维, V1 简化版)
    输出: 128维特征向量
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]  # dynamically matches env (6 for V1)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, features_dim),
            nn.ELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# SAC 的策略关键字参数，传给 SAC(policy_kwargs=...)
POLICY_KWARGS = dict(
    features_extractor_class=ResidualFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(
        pi=[256, 256],   # Actor 隐藏层
        qf=[256, 256],   # Critic 隐藏层
    ),
    activation_fn=nn.ELU,
)