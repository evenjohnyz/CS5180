import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer

env = gym.make("PandaPickAndPlace-v3")
log_dir = './panda_reach_v3_tensorboard/'

# DDPG
model = DDPG(policy="MultiInputPolicy", env=env, buffer_size=1000000, replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=200000)
model.save("ddpg_panda_reach_v3")

