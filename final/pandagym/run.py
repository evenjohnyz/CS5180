# run.py (updated version)
import argparse
import torch
import numpy as np
import os
import panda_gym  
import gymnasium as gym

from environments.wrappers import PandaEnvWrapper
from utils.buffer import ReplayBuffer
from utils.her_buffer import HERBuffer
from algorithms.ddpg import DDPG
from algorithms.sac import SAC
from algorithms.td3 import TD3
from experiments.train import train_agent
from experiments.train_her import train_with_her
from experiments.evaluate import evaluate_trained_agent
from config.default import ddpg_params, sac_params, td3_params

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and evaluate RL agents on Panda-Gym")
    
    # Environment parameters
    parser.add_argument("--env", type=str, default="PandaReach-v3", 
                        help="Panda-Gym environment name (use -Dense suffix for dense rewards)")
    
    # Algorithm parameters
    parser.add_argument("--algorithm", type=str, default="DDPG", 
                        choices=["DDPG", "SAC", "TD3"], help="RL algorithm")
    parser.add_argument("--use_her", action="store_true", 
                        help="Use Hindsight Experience Replay")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_timesteps", type=int, default=200000,
                        help="Maximum number of timesteps")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for training")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--load_model", type=str, default="", help="Load model from path")
    
    # Rendering
    parser.add_argument("--render", action="store_true", help="Render environment")
    
    # HER specific parameters
    parser.add_argument("--k_future", type=int, default=4,
                        help="Number of future goals to sample in HER")
    parser.add_argument("--strategy", type=str, default="future",
                        choices=["future", "episode", "random"], 
                        help="Goal sampling strategy in HER")
    
    # Algorithm specific parameters
    # SAC
    parser.add_argument("--alpha", type=float, default=0.2, 
                        help="Initial entropy coefficient (SAC)")
    parser.add_argument("--auto_entropy", action="store_true", 
                        help="Use automatic entropy tuning (SAC)")
    # TD3
    parser.add_argument("--policy_noise", type=float, default=0.2, 
                        help="Policy noise for exploration (TD3)")
    parser.add_argument("--noise_clip", type=float, default=0.5, 
                        help="Noise clip for exploration (TD3)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if using dense rewards (based on environment name)
    is_dense = "Dense" in args.env
    
    # Create directory for results
    her_suffix = "_HER" if args.use_her else ""
    dense_suffix = "_dense" if is_dense else "_sparse"
    save_dir = f"results/{args.algorithm}{her_suffix}_{args.env}_seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = PandaEnvWrapper(
        env_name=args.env,
        render_mode="human" if args.render else "rgb_array",
        use_her=args.use_her
    )

    from gymnasium.wrappers import TimeLimit
    env.env = TimeLimit(env.env.unwrapped, max_episode_steps=200)
    
    print(f"Created environment: {args.env}")
    if args.use_her:
        print(f"State dim: {env.state_dim}, Goal dim: {env.goal_dim}, Action dim: {env.action_dim}")
    else:
        print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    
    # If using HER, adjust state dimension to include goal
    state_dim = env.state_dim + env.goal_dim if args.use_her else env.state_dim
    
    # Create agent based on algorithm
    if args.algorithm == "DDPG":
        agent = DDPG(
            state_dim=state_dim,
            action_dim=env.action_dim,
            max_action=env.max_action,
            device=device
        )
    elif args.algorithm == "SAC":
        agent = SAC(
            state_dim=state_dim,
            action_dim=env.action_dim,
            max_action=env.max_action,
            alpha=args.alpha,
            automatic_entropy_tuning=args.auto_entropy,
            device=device
        )
    elif args.algorithm == "TD3":
        agent = TD3(
            state_dim=state_dim,
            action_dim=env.action_dim,
            max_action=env.max_action,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            device=device
        )
    
    # Create buffer
    if args.use_her:
        buffer = HERBuffer(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            goal_dim=env.goal_dim,
            max_size=1e6,
            k_future=args.k_future,
            strategy=args.strategy
        )
    else:
        buffer = ReplayBuffer(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            max_size=1e6
        )
    
    # Set training arguments
    train_args = argparse.Namespace(
        algorithm=args.algorithm,
        env=args.env,
        max_timesteps=args.max_timesteps,
        start_timesteps=1000,  # Random exploration steps
        batch_size=args.batch_size,
        eval_freq=10000,
        eval_episodes=10,
        expl_noise=0.1,
        max_episode_steps=1000,
        save_dir=save_dir,
        device=device
    )
    
    # Set evaluation arguments
    eval_args = argparse.Namespace(
        algorithm=args.algorithm,
        env=args.env,
        eval_episodes=20,
        max_episode_steps=1000,
        render=args.render,
        save_dir=save_dir,
        load_model=args.load_model
    )
    
    # Run training or evaluation
    if not args.evaluate_only:
        print(f"Starting training for {args.max_timesteps} timesteps...")
        if args.use_her:
            train_stats = train_with_her(env, agent, buffer, train_args)
        else:
            train_stats = train_agent(env, agent, buffer, train_args)
        print("Training completed!")
    
    if args.evaluate_only or not args.evaluate_only:
        print("Starting evaluation...")
        eval_stats = evaluate_trained_agent(env, agent, eval_args)
        print("Evaluation completed!")
    
    env.close()