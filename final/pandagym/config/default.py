# config/default.py
"""Default configuration for RL algorithms."""

# Common parameters for all algorithms
common_params = {
    # Environment
    "env": "PandaPickAndPlace-v3",
    "reward_type": "sparse",
    "max_episode_steps": 1000,
    
    # Training
    "seed": 0,
    "max_timesteps": 1000000,
    "start_timesteps": 1000,
    "batch_size": 256,
    "eval_freq": 5000,
    "eval_episodes": 10,
    "expl_noise": 0.1,
    
    # Replay buffer
    "buffer_size": 1000000,
    
    # Saving and rendering
    "save_dir": "results",
    "render": False,
}

# DDPG specific parameters
ddpg_params = {
    **common_params,
    "algorithm": "DDPG",
    "discount": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
}

# SAC specific parameters
sac_params = {
    **common_params,
    "algorithm": "SAC",
    "discount": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
}

# TD3 specific parameters
td3_params = {
    **common_params,
    "algorithm": "TD3",
    "discount": 0.99,
    "tau": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_freq": 2,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
}