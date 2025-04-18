import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from torch import nn

# Create the environment
env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
log_dir = './log/'

# The noise objects for TD3 (Scale of additive gaussian noise 0.1)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

# Create and configure TD3 model with HER
model = TD3(
    policy="MultiInputPolicy",  # Important: use MultiInputPolicy for Dict observation spaces
    env=env,
    learning_rate=5e-4,
    buffer_size=1_000_000,  # Replay buffer size
    learning_starts=1000,   # How many steps before learning starts
    batch_size=256,
    tau=0.05,              # Soft update coefficient for target networks
    gamma=0.95,             # Discount factor
    action_noise=action_noise,
    # TD3-specific parameters
    policy_delay=3,                # Delay policy updates compared to critic updates
    target_policy_noise=0.2,       # Noise added to target policy
    target_noise_clip=0.5,         # Clipping of target policy noise
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

model.learn(total_timesteps=2_000_000)

# save model
model.save("models/td3_her_panda_pick_and_place")

# test model
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()