# experiments/train_her.py (updated version)
import numpy as np
import torch
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_with_her(env, agent, her_buffer, args):
    """
    Train an agent with Hindsight Experience Replay.
    
    Args:
        env: HER-compatible environment wrapper.
        agent: RL agent to train.
        her_buffer: HER replay buffer.
        args: Training arguments and hyperparameters.
        
    Returns:
        dict: Training statistics.
    """
    # Check if environment is compatible with HER
    if not hasattr(env, 'use_her') or not env.use_her:
        raise ValueError("Environment must be initialized with use_her=True for HER training")
    
    print(f"Training with max_episode_steps = {args.max_episode_steps}")
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    evaluation_scores = []
    training_time = 0
    
    # Create directories for saving results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Training loop
    total_timesteps = 0
    
    # Progress bar
    progress_bar = tqdm(total=args.max_timesteps, desc=f"Training {args.algorithm} with HER")
    
    while total_timesteps < args.max_timesteps:
        episode_reward = 0
        episode_length = 0
        episode_success = 0
        done = False
        truncated = False
        
        # Reset the environment
        state, achieved_goal, desired_goal, _ = env.reset()
        
        # Episode loop
        while not (done or truncated) and episode_length < args.max_episode_steps:
            # Select action
            if total_timesteps < args.start_timesteps:
                # Random action for initial exploration
                action = env.action_space.sample()
            else:
                # Agent selects action
                # Concatenate state and goal for goal-conditioned policy
                goal_state = np.concatenate([state, desired_goal])
                action = agent.select_action(goal_state)
                if args.expl_noise != 0:
                    noise = np.random.normal(0, args.expl_noise, size=env.action_dim)
                    action = np.clip(action + noise, -env.max_action, env.max_action)
            
            # Take step in environment
            next_state, next_achieved_goal, desired_goal, reward, done, truncated, info = env.step(action)
            
            # Store transition in buffer
            her_buffer.add(
                state=state,
                action=action,
                next_state=next_state,
                goal=desired_goal,
                achieved_goal=achieved_goal,
                next_achieved_goal=next_achieved_goal,
                reward=reward,
                done=float(done)
            )
            
            # Update current state and stats
            state = next_state
            achieved_goal = next_achieved_goal
            episode_reward += reward
            episode_length += 1
            total_timesteps += 1
            progress_bar.update(1)
            
            # Check if episode was successful
            if 'is_success' in info:
                episode_success = max(episode_success, info['is_success'])
            
            # Train agent
            if total_timesteps >= args.start_timesteps:
                train_start = time.time()
                
                # Sample batch with HER
                states, actions, next_states, goals, rewards, not_dones = her_buffer.sample(args.batch_size)
                
                # Concatenate states and goals
                goal_states = np.concatenate([states, goals], axis=1)
                goal_next_states = np.concatenate([next_states, goals], axis=1)
                
                # Convert to tensors
                goal_states = torch.FloatTensor(goal_states).to(args.device)
                actions = torch.FloatTensor(actions).to(args.device)
                goal_next_states = torch.FloatTensor(goal_next_states).to(args.device)
                rewards = torch.FloatTensor(rewards).to(args.device)
                not_dones = torch.FloatTensor(not_dones).to(args.device)
                
                # Update agent
                train_metrics = agent.train_on_batch(
                    goal_states, actions, goal_next_states, rewards, not_dones
                )
                
                training_time += time.time() - train_start
            
            # Check if we've reached max timesteps
            if total_timesteps >= args.max_timesteps:
                break
        
        # Episode completed - store in HER buffer
        her_buffer.store_episode()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rates.append(episode_success)
        
        # Print episode stats
        if len(episode_rewards) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_success = np.mean(success_rates[-10:])
            print(f"\nEpisode {len(episode_rewards)}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}, Success Rate = {avg_success:.2f}")
        
        # Evaluate agent periodically
        if total_timesteps % args.eval_freq == 0:
            eval_success_rate, eval_reward = evaluate_her_agent(env, agent, args.eval_episodes)
            evaluation_scores.append((total_timesteps, eval_reward, eval_success_rate))
            
            # Save the model if it's the best so far
            if len(evaluation_scores) == 1 or eval_success_rate > max([s for _, _, s in evaluation_scores[:-1]]):
                agent.save(f"{args.save_dir}/{args.algorithm}_her_best")
            
            # Save checkpoint
            agent.save(f"{args.save_dir}/{args.algorithm}_her_{total_timesteps}")
            
            print(f"\nTime step {total_timesteps}: Evaluation reward = {eval_reward:.2f}, Success rate = {eval_success_rate:.2f}")
    
    progress_bar.close()
    
    # Final evaluation
    final_success_rate, final_reward = evaluate_her_agent(env, agent, args.eval_episodes)
    print(f"\nTraining completed. Final evaluation: Reward = {final_reward:.2f}, Success rate = {final_success_rate:.2f}")
    
    # Save final agent
    agent.save(f"{args.save_dir}/{args.algorithm}_her_final")
    
    # Save training statistics
    stats = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rates": success_rates,
        "evaluation_scores": evaluation_scores,
        "training_time": training_time,
        "final_success_rate": final_success_rate,
        "final_reward": final_reward
    }
    
    # Plot learning curve
    plot_her_learning_curve(stats, args)
    
    return stats

def evaluate_her_agent(env, agent, num_episodes=10):
    """
    Evaluate agent with HER.
    
    Args:
        env: HER-compatible environment.
        agent: Agent to evaluate.
        num_episodes: Number of episodes to evaluate.
        
    Returns:
        tuple: (success_rate, average_reward)
    """
    success_count = 0
    total_rewards = []
    
    for _ in range(num_episodes):
        episode_reward = 0
        episode_success = 0
        
        state, achieved_goal, desired_goal, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Concatenate state and goal
            goal_state = np.concatenate([state, desired_goal])
            
            # Select action (no exploration noise for evaluation)
            action = agent.select_action(goal_state, evaluate=True)
            
            # Step environment
            next_state, next_achieved_goal, desired_goal, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # Check for success
            if 'is_success' in info:
                episode_success = max(episode_success, info['is_success'])
            
            state = next_state
            achieved_goal = next_achieved_goal
        
        success_count += episode_success
        total_rewards.append(episode_reward)
    
    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    
    return success_rate, avg_reward

def plot_her_learning_curve(stats, args):
    """
    Plot learning curves for HER training.
    
    Args:
        stats (dict): Training statistics.
        args: Training arguments.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot episode rewards
    ax1.plot(stats["episode_rewards"], label="Episode Reward")
    # Add moving average
    window_size = min(10, len(stats["episode_rewards"]))
    if window_size > 0:
        moving_avg = np.convolve(stats["episode_rewards"], 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        ax1.plot(np.arange(window_size-1, len(stats["episode_rewards"])), 
                 moving_avg, 
                 'r', 
                 label=f"{window_size}-Episode Moving Average")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward")
    ax1.set_title(f"Episode Rewards for {args.algorithm} with HER")
    ax1.legend()
    ax1.grid(True)
    
    # Plot success rates
    ax2.plot(stats["success_rates"], label="Success Rate")
    # Add moving average
    window_size = min(10, len(stats["success_rates"]))
    if window_size > 0:
        moving_avg = np.convolve(stats["success_rates"], 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        ax2.plot(np.arange(window_size-1, len(stats["success_rates"])), 
                 moving_avg, 
                 'r', 
                 label=f"{window_size}-Episode Moving Average")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Success Rate")
    ax2.set_title(f"Success Rates for {args.algorithm} with HER")
    ax2.legend()
    ax2.grid(True)
    
    # Plot evaluation scores
    timesteps, rewards, success_rates = zip(*stats["evaluation_scores"])
    ax3.plot(timesteps, rewards, 'b-o', label="Evaluation Reward")
    ax3.set_xlabel("Timesteps")
    ax3.set_ylabel("Average Reward")
    ax3.set_title(f"Evaluation Scores for {args.algorithm} with HER")
    ax3.legend()
    ax3.grid(True)
    
    # Add success rate on secondary y-axis
    ax3_twin = ax3.twinx()
    ax3_twin.plot(timesteps, success_rates, 'g-o', label="Success Rate")
    ax3_twin.set_ylabel("Success Rate")
    ax3_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/{args.algorithm}_her_learning_curve.png")
    plt.close()