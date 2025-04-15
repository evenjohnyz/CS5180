# experiments/evaluate.py
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_trained_agent(env, agent, args):
    """
    Comprehensive evaluation of a trained agent.
    
    Args:
        env: Environment to evaluate in.
        agent: Trained agent to evaluate.
        args: Evaluation arguments.
        
    Returns:
        dict: Evaluation statistics.
    """
    # Load agent if path is provided
    if args.load_model:
        agent.load(args.load_model)
        print(f"Loaded agent from {args.load_model}")
    
    # Statistics
    success_rate = 0
    episode_rewards = []
    episode_lengths = []
    completion_times = []
    
    # Visualization setup
    if args.render:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Run evaluation episodes
    for episode in tqdm(range(args.eval_episodes), desc="Evaluating"):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False
        start_time = time.time()
        
        # Episode loop
        while not (done or truncated):
            # Select action without noise
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            # Break if episode is too long
            if episode_steps >= args.max_episode_steps:
                break
        
        # Record episode statistics
        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        completion_times.append(episode_time)
        
        # Check if task was successful (based on final reward or info)
        if 'is_success' in info:
            success_rate += info['is_success']
        elif reward > 0:  # Assuming positive reward indicates success in sparse reward setting
            success_rate += 1
    
    # Calculate final statistics
    success_rate = success_rate / args.eval_episodes
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_time = np.mean(completion_times)
    
    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f} steps")
    print(f"Average Completion Time: {avg_time:.4f} seconds")
    
    # Visualization of results
    plot_evaluation_results(episode_rewards, episode_lengths, args)
    
    # Save statistics
    stats = {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "avg_time": avg_time,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "completion_times": completion_times
    }
    
    return stats

def plot_evaluation_results(rewards, lengths, args):
    """
    Plot histograms of evaluation results.
    
    Args:
        rewards (list): Episode rewards.
        lengths (list): Episode lengths.
        args: Evaluation arguments.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward histogram
    ax1.hist(rewards, bins=10, alpha=0.7)
    ax1.axvline(np.mean(rewards), color='r', linestyle='dashed', linewidth=1)
    ax1.set_xlabel("Episode Reward")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Reward Distribution ({args.algorithm})")
    
    # Length histogram
    ax2.hist(lengths, bins=10, alpha=0.7)
    ax2.axvline(np.mean(lengths), color='r', linestyle='dashed', linewidth=1)
    ax2.set_xlabel("Episode Length (steps)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Episode Length Distribution ({args.algorithm})")
    
    plt.tight_layout()
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(f"{args.save_dir}/{args.algorithm}_evaluation_results.png")
    plt.close()