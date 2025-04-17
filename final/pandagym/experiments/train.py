# experiments/train.py (updated version)
import numpy as np
import torch
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_agent(env, agent, replay_buffer, args):
    """
    Train an agent in the given environment using the specified parameters.
    
    Args:
        env: Environment wrapper.
        agent: RL agent to train.
        replay_buffer: Experience replay buffer.
        args: Training arguments and hyperparameters.
        
    Returns:
        dict: Training statistics.
    """
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    episode_success_rates = []  # trace success rate 
    evaluation_scores = []
    evaluation_success_rates = []  # trave evaluation of success rate
    training_time = 0
    
    # Create directories for saving results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Training loop
    total_timesteps = 0
    episode_count = 0  # add counter
    

    should_exit = False
    
    # Progress bar
    progress_bar = tqdm(total=args.max_timesteps, desc=f"Training {args.algorithm}")
    
    while total_timesteps < args.max_timesteps and not should_exit:
        episode_count += 1  # start counting 
        episode_reward = 0
        episode_length = 0
        episode_success = 0  
        done = False
        truncated = False
        
        # Reset the environment
        state, info = env.reset()
        
        while not (done or truncated) and episode_length < args.max_episode_steps:
            # Select action with noise for exploration
            if total_timesteps < args.start_timesteps:
                # Random action for initial exploration
                action = env.action_space.sample()
            else:
                # Agent selects action with exploration noise
                action = agent.select_action(state)
                if args.expl_noise != 0:
                    noise = np.random.normal(0, args.expl_noise, size=env.action_dim)
                    action = np.clip(action + noise, -env.max_action, env.max_action)
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, next_state, reward, float(done))
            
            # Update current state and stats
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_timesteps += 1
            progress_bar.update(1)
            
            # update success signal
            if 'is_success' in info:
                episode_success = max(episode_success, info['is_success'])
            elif reward > 0.9:  # update threshould value depoends on the env
                episode_success = 1.0
            
            # Train agent
            if total_timesteps >= args.start_timesteps:
                train_start = time.time()
                train_metrics = agent.train(replay_buffer, args.batch_size)
                training_time += time.time() - train_start

            # THIS IS EVALUATION BASED ON TIMESTEP
            # if total_timesteps % args.eval_freq == 0:
            #     eval_score, eval_success = evaluate_agent(env, agent, args.eval_episodes)
            #     evaluation_scores.append((total_timesteps, eval_score))
            #     evaluation_success_rates.append((total_timesteps, eval_success))
            #     
            #     # Save the model if it's the best so far
            #     if len(evaluation_scores) == 1 or eval_score > max([s for _, s in evaluation_scores[:-1]]):
            #         agent.save(f"{args.save_dir}/{args.algorithm}_best")
            #     
            #     # Save checkpoint
            #     agent.save(f"{args.save_dir}/{args.algorithm}_{total_timesteps}")
            #     
            #     print(f"\nTime step {total_timesteps}: Evaluation score = {eval_score:.2f}, Success rate = {eval_success:.2f}")
            
            # Check if we've reached max timesteps
            if total_timesteps >= args.max_timesteps:
                should_exit = True
                break
        
        # Episode completed
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_success_rates.append(episode_success)  # record if success or not
        
        # evaluate for 4 steps
        if episode_count % 4 == 0:
            eval_score, eval_success = evaluate_agent(env, agent, args.eval_episodes)
            evaluation_scores.append((total_timesteps, eval_score))
            evaluation_success_rates.append((total_timesteps, eval_success))
            
            # Save the model if it's the best so far
            if len(evaluation_scores) == 1 or eval_score > max([s for _, s in evaluation_scores[:-1]]):
                agent.save(f"{args.save_dir}/{args.algorithm}_best")
            
            # Save checkpoint
            agent.save(f"{args.save_dir}/{args.algorithm}_{total_timesteps}")
            
            print(f"\nEpisode {episode_count}, Time step {total_timesteps}: Evaluation score = {eval_score:.2f}, Success rate = {eval_success:.2f}")
        
        # Print episode stats
        if len(episode_rewards) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_success = np.mean(episode_success_rates[-10:])  # caculate success rate for recent 10 
            print(f"\nEpisode {len(episode_rewards)}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}, Success Rate = {avg_success:.2f}")
    
    progress_bar.close()
    
    # Final evaluation
    final_score, final_success = evaluate_agent(env, agent, args.eval_episodes)
    print(f"\nTraining completed. Final evaluation score: {final_score:.2f}, Success rate: {final_success:.2f}")
    
    # Save final agent
    agent.save(f"{args.save_dir}/{args.algorithm}_final")
    
    # Save training statistics
    stats = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_success_rates": episode_success_rates,  # add success rate 
        "evaluation_scores": evaluation_scores,
        "evaluation_success_rates": evaluation_success_rates,  # add evaluation of success rate
        "training_time": training_time,
        "final_score": final_score,
        "final_success_rate": final_success  # add final success rate
    }
    
    # Plot learning curve with success rate
    plot_learning_curve(stats, args)
    
    return stats

def evaluate_agent(env, agent, num_episodes=10):
    """
    Evaluate the agent's performance without exploration noise.
    
    Args:
        env: Environment to evaluate in.
        agent: Agent to evaluate.
        num_episodes: Number of episodes to evaluate.
        
    Returns:
        tuple: (average_reward, success_rate) 返回值增加成功率
    """
    eval_rewards = []
    successes = 0  # record number of success
    
    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_success = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action without noise
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            # check is success
            if 'is_success' in info:
                episode_success = max(episode_success, info['is_success'])
            elif reward > 0.9:  
                episode_success = 1.0
        
        eval_rewards.append(episode_reward)
        successes += episode_success
    
    success_rate = successes / num_episodes
    return np.mean(eval_rewards), success_rate

def plot_learning_curve(stats, args):
    """
    Plot and save the learning curves with enhanced styling similar to the example.
    
    Args:
        stats (dict): Training statistics.
        args: Training arguments.
    """
    # create multiple plots
    
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    # success rate
    timesteps, scores = zip(*stats["evaluation_scores"])
    timesteps_success, success_rates = zip(*stats["evaluation_success_rates"])
    
    timesteps_k = [t/1000 for t in timesteps_success]
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    window_size = 20 
    episode_successes = stats["episode_success_rates"]
    
    # caculate episode length for each time step
    total_steps = args.max_timesteps
    total_episodes = len(episode_successes)
    steps_per_episode = [args.max_episode_steps] * total_episodes
    episode_steps = np.cumsum(steps_per_episode)
    
    valid_indices = episode_steps <= total_steps
    episode_steps = episode_steps[valid_indices]
    episode_successes = episode_successes[:len(episode_steps)]
    
    episode_steps_k = [s/1000 for s in episode_steps]
    
    # smoothing the success rate plot
    if len(episode_successes) > window_size:
        smooth_successes = []
        for i in range(len(episode_successes) - window_size + 1):
            smooth_successes.append(np.mean(episode_successes[i:i+window_size]))
        smooth_x = episode_steps_k[window_size-1:]
        
        plt.plot(smooth_x, smooth_successes, color='#40E0D0', linewidth=1.5, label=f'Training (smoothed)')
    
    plt.plot(timesteps_k, success_rates, color='#1E90FF', linewidth=2, label='Evaluation')
    
    # horizontal line
    for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
        plt.axhline(y=y, color='gray', linestyle='-', alpha=0.3)
    
    plt.ylim(0, 1.05)
    plt.xlim(0, max(timesteps_k))
    plt.xlabel('Timesteps (K)', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title(f'Success Rate for {args.algorithm} on {args.env}', fontsize=14)
    
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/{args.algorithm}_success_rate.png", dpi=300)
    plt.style.use('default')
    

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
    
    # reward plot
    ax1.plot(stats["episode_rewards"], label="Episode Reward")
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
    ax1.set_title(f"Episode Rewards for {args.algorithm} on {args.env}")
    ax1.legend()
    ax1.grid(True)
    
    # success plot
    ax2.plot(stats["episode_success_rates"], label="Success Rate")
    window_size = min(10, len(stats["episode_success_rates"]))
    if window_size > 0:
        moving_avg = np.convolve(stats["episode_success_rates"], 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        ax2.plot(np.arange(window_size-1, len(stats["episode_success_rates"])), 
                 moving_avg, 
                 'r', 
                 label=f"{window_size}-Episode Moving Average")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Success Rate")
    ax2.set_title(f"Episode Success Rates for {args.algorithm} on {args.env}")
    ax2.legend()
    ax2.grid(True)
    
    # evaluation plot
    ax3.plot(timesteps, scores, 'b-o', label="Evaluation Score")
    ax3.set_xlabel("Timesteps")
    ax3.set_ylabel("Average Reward")
    ax3.set_title(f"Evaluation Scores for {args.algorithm} on {args.env}")
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(timesteps_success, success_rates, 'g-o', label="Success Rate")
    ax3_twin.set_ylabel("Success Rate")
    ax3_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/{args.algorithm}_all_metrics.png")
    plt.close('all')