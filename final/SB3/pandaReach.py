import gymnasium as gym
import panda_gym
import numpy as np
import torch
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy

# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# 创建环境
env = gym.make("PandaReach-v3")
env.reset(seed=seed)
env.action_space.seed(seed)

log_dir = './panda_reach_v3_tensorboard/'

# 评估函数
def train_and_evaluate(model_class, name, use_her=False, n_critics=None, buffer_size=1_000_000, total_timesteps=30000):
    print(f"\nTraining {name}...")
    kwargs = dict(
        policy="MultiInputPolicy",
        env=env,
        buffer_size=buffer_size,
        verbose=1,
        tensorboard_log=log_dir
    )

    if use_her:
        kwargs["replay_buffer_class"] = HerReplayBuffer
    if n_critics is not None:
        kwargs["policy_kwargs"] = dict(n_critics=n_critics)

    model = model_class(**kwargs)
    model.learn(total_timesteps=total_timesteps)

    # 保存模型
    model.save(name)

    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"{name} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# ===== Basic (non-HER) =====
train_and_evaluate(DDPG, "ddpg_panda_reach_basic")
train_and_evaluate(TD3,  "td3_panda_reach_basic")
train_and_evaluate(SAC,  "sac_panda_reach_basic")

# ===== With HER =====
train_and_evaluate(DDPG, "ddpg_panda_reach_her", use_her=True)
train_and_evaluate(TD3,  "td3_panda_reach_her", use_her=True)
train_and_evaluate(SAC,  "sac_panda_reach_her", use_her=True)

# ===== HER + Single Critic (No double Q) =====
train_and_evaluate(TD3,  "td3_panda_reach_her_singleq", use_her=True, n_critics=1)
train_and_evaluate(SAC,  "sac_panda_reach_her_singleq", use_her=True, n_critics=1)
