# utils/her_buffer.py
import numpy as np
import random

class HERBuffer:
    """
    Hindsight Experience Replay buffer.
    Stores transitions and goals for goal-conditioned reinforcement learning.
    """
    def __init__(self, state_dim, action_dim, goal_dim, max_size=1e6, k_future=4, strategy='future'):
        """
        Initialize HER buffer.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            goal_dim (int): Dimension of goal space.
            max_size (int): Maximum size of the buffer.
            k_future (int): Number of future goals to sample.
            strategy (str): Goal sampling strategy ('future', 'episode', 'random').
        """
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        
        self.k_future = k_future
        self.strategy = strategy
        self.goal_dim = goal_dim
        
        # Buffer storage
        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.goal = np.zeros((self.max_size, goal_dim))
        self.achieved_goal = np.zeros((self.max_size, goal_dim))
        self.next_achieved_goal = np.zeros((self.max_size, goal_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        
        # Episode storage for HER
        self.episode_start_indices = []
        self.current_episode_start = 0
        
    def add(self, state, action, next_state, goal, achieved_goal, next_achieved_goal, reward, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            next_state: Next state.
            goal: Desired goal.
            achieved_goal: Current achieved goal.
            next_achieved_goal: Next achieved goal.
            reward: Reward received.
            done: Whether the episode has ended.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.goal[self.ptr] = goal
        self.achieved_goal[self.ptr] = achieved_goal
        self.next_achieved_goal[self.ptr] = next_achieved_goal
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def store_episode(self):
        """
        Store the current episode end index.
        Call this at the end of each episode.
        """
        episode_length = self.ptr - self.current_episode_start
        if self.current_episode_start + episode_length >= self.max_size:
            episode_length = self.max_size - self.current_episode_start
            
        if episode_length > 0:
            self.episode_start_indices.append((self.current_episode_start, episode_length))
        
        # Update current episode start
        self.current_episode_start = self.ptr
            
    def _sample_achieved_goals(self, episode_start, episode_length, t, n_samples):
        """
        Sample achieved goals from the episode according to strategy.
        
        Args:
            episode_start (int): Start index of the episode.
            episode_length (int): Length of the episode.
            t (int): Current time step in the episode.
            n_samples (int): Number of goals to sample.
            
        Returns:
            np.ndarray: Sampled achieved goals.
        """
        if self.strategy == 'future':
            # 计算有效的未来时间步范围
            if episode_length - t - 1 > 0:
                # 使用numpy直接生成随机数组，避免列表推导式
                offsets = np.random.randint(1, episode_length - t, size=n_samples)
                future_indices = (episode_start + t + offsets) % self.max_size
            else:
                # 如果没有未来时间步，从整个回合采样
                offsets = np.random.randint(0, episode_length, size=n_samples)
                future_indices = (episode_start + offsets) % self.max_size
            
            return self.achieved_goal[future_indices]
        
        elif self.strategy == 'episode':
            # 从整个回合中随机采样
            offsets = np.random.randint(0, episode_length, size=n_samples)
            episode_indices = (episode_start + offsets) % self.max_size
            
            return self.achieved_goal[episode_indices]
        
        elif self.strategy == 'random':
            # 从整个缓冲区随机采样
            random_indices = np.random.randint(0, self.size, size=n_samples)
            
            return self.achieved_goal[random_indices]
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def compute_reward(self, achieved_goal, goal):
        """
        Compute reward for reaching a goal.
        This is a placeholder - the actual reward function depends on the environment.
        
        Args:
            achieved_goal: The achieved goal.
            goal: The desired goal.
            
        Returns:
            float: Reward value.
        """
        # Typical sparse reward function for reaching a goal
        # Success if distance between achieved goal and goal is below a threshold
        threshold = 0.05
        distance = np.linalg.norm(achieved_goal - goal, axis=-1)
        return (distance < threshold).astype(np.float32).reshape(-1, 1)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions with HER.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            tuple: Batch of (states, actions, next_states, goals, rewards, not_dones)
        """
        # Sample indices
        if self.size < batch_size:
            # If not enough samples, repeat some
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            indices = np.random.permutation(self.size)[:batch_size]
        
        # Get batch of original transitions
        states = self.state[indices]
        actions = self.action[indices]
        next_states = self.next_state[indices]
        goals = self.goal[indices]
        rewards = self.reward[indices]
        not_dones = self.not_done[indices]
        
        # Apply HER to some transitions
        her_indices = np.random.permutation(batch_size)[:batch_size // 2]  # Use HER for half the batch
        
        for i in her_indices:
            # Find which episode this transition belongs to
            original_idx = indices[i]
            
            # Find the episode
            episode_found = False
            for start_idx, length in self.episode_start_indices:
                if start_idx <= original_idx < start_idx + length:
                    episode_start = start_idx
                    episode_length = length
                    t = original_idx - start_idx
                    episode_found = True
                    break
            
            if not episode_found:
                continue
            
            # Sample alternative goal
            alternative_goals = self._sample_achieved_goals(episode_start, episode_length, t, 1)
            alternative_goal = alternative_goals[0]
            
            # Compute new reward with alternative goal
            new_reward = self.compute_reward(self.next_achieved_goal[original_idx], alternative_goal)
            
            # Update the transition with the alternative goal
            goals[i] = alternative_goal
            rewards[i] = new_reward
            
            # If goal is reached, mark as done
            if new_reward > 0:
                not_dones[i] = 0.0
        
        return states, actions, next_states, goals, rewards, not_dones