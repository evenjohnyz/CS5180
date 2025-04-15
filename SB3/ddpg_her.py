import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from torch import nn
import torch as th


# Create the environment
env = gym.make("PandaReach-v3")
log_dir = './pandaReach_log/'

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

# Create and configure DDPG model with HER
model = DDPG(
    policy="MultiInputPolicy",  # Important: use MultiInputPolicy for Dict observation spaces
    env=env,
    learning_rate=1e-3,
    buffer_size=1_000_000,  # Replay buffer size
    learning_starts=1000,   # How many steps before learning starts
    batch_size=256,
    tau=0.05,              # Soft update coefficient for target networks
    gamma=0.99,             # Discount factor
    action_noise=action_noise,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,    # Number of goals sampled per actual goal
        goal_selection_strategy="future",  # Sample goals from future timesteps in episode
    ),
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]),  # Neural network architecture
        optimizer_kwargs=dict(weight_decay=1.0)
    ),
    tensorboard_log=log_dir,
    verbose=1
)


model.learn(total_timesteps=200_000)

# Save the model
model.save("models/ddpg_her_pandaReach")

# Test the model
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()