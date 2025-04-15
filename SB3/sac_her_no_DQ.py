import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from torch import nn

# Create the environment
env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
log_dir = './log/NO_DQ/'

# Create and configure SAC model with HER
model = SAC(
    policy="MultiInputPolicy",  # Important: use MultiInputPolicy for Dict observation spaces
    env=env,
    learning_rate=1e-3,
    buffer_size=1_000_000,  # Replay buffer size
    learning_starts=1000,   # How many steps before learning starts
    batch_size=256,
    tau=0.005,              # Soft update coefficient for target networks
    gamma=0.95,             # Discount factor
    ent_coef='auto',        # Entropy coefficient (auto-adjusting)
    target_update_interval=1,
    train_freq=1,           # Update the model every step
    gradient_steps=1,       # How many gradient updates after each step
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,    # Number of goals sampled per actual goal
        goal_selection_strategy="future",  # Sample goals from future timesteps in episode

    ),
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]),  # Neural network architecture
        n_critics=1
    ),
    tensorboard_log=log_dir,
    verbose=1
)

# Train the model - 测试每80个情节进行一次
model.learn(total_timesteps=2_000_000)

# Save the model
model.save("models/sac_her_No_DQ_panda_pick_and_place")

# Test the model
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()