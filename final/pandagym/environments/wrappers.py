# environments/wrappers.py (updated version)
import gymnasium as gym
import numpy as np

class PandaEnvWrapper:
    """
    Wrapper for Panda-Gym environments to provide a consistent interface.
    Supports both standard mode and HER mode.
    """
    def __init__(self, env_name='PandaPickAndPlace-v3', render_mode='human', use_her=False):
        """
        Initialize the environment wrapper.
        
        Args:
            env_name (str): Name of the Panda-Gym environment.
            render_mode (str): Rendering mode ('human', 'rgb_array', or None).
            use_her (bool): Whether to use Hindsight Experience Replay.
        """
        # Fix: If render_mode is None, default to 'rgb_array'
        if render_mode is None:
            render_mode = 'rgb_array'
            
        self.env = gym.make(env_name, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_mode = render_mode
        self.use_her = use_her
        
        # Check if environment supports goals (for HER)
        self.is_goal_env = isinstance(self.observation_space, gym.spaces.Dict) and 'desired_goal' in self.observation_space.spaces
        
        if use_her and not self.is_goal_env:
            raise ValueError(f"Environment {env_name} does not seem to be a goal-based environment")
        
        # Extract dimension information
        if use_her and self.is_goal_env:
            self.state_dim = self._get_dim(self.observation_space.spaces['observation'])
            self.goal_dim = self._get_dim(self.observation_space.spaces['desired_goal'])
            self.action_dim = self.action_space.shape[0]
        else:
            self.state_dim = self.get_state_dim()
            self.action_dim = self.action_space.shape[0]
            self.goal_dim = 0  # No goal dimension in non-HER mode
            
        self.max_action = self.action_space.high[0]
        
    def _get_dim(self, space):
        """Get the dimension of a space."""
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    
    def get_state_dim(self):
        """Get the dimension of the state space."""
        if isinstance(self.observation_space, gym.spaces.Dict):
            # If observation is a dictionary, flatten it
            space_dict = self.observation_space.spaces
            state_dim = sum(np.prod(space.shape) for space in space_dict.values())
            return int(state_dim)
        else:
            # If observation is a Box, just return its shape
            return int(np.prod(self.observation_space.shape))
    
    def normalize_observation(self, obs):
        """
        Normalize and flatten the observation if needed.
        
        Args:
            obs: The observation from the environment.
            
        Returns:
            np.ndarray: Flattened and normalized observation.
        """
        if isinstance(obs, dict):
            # If observation is a dictionary, flatten it
            return np.concatenate([obs[k].ravel() for k in sorted(obs.keys())])
        return obs
    
    def reset(self):
        """Reset the environment."""
        obs, info = self.env.reset()
        
        if self.use_her and self.is_goal_env:
            # HER mode: separate observation, achieved goal and desired goal
            observation = obs['observation']
            achieved_goal = obs['achieved_goal']
            desired_goal = obs['desired_goal']
            return observation, achieved_goal, desired_goal, info
        else:
            # Standard mode: flatten observation
            return self.normalize_observation(obs), info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Action to take.
            
        Returns:
            tuple: Different tuples depending on the mode.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.use_her and self.is_goal_env:
            # HER mode
            observation = obs['observation']
            achieved_goal = obs['achieved_goal']
            desired_goal = obs['desired_goal']
            return observation, achieved_goal, desired_goal, reward, terminated, truncated, info
        else:
            # Standard mode
            return self.normalize_observation(obs), reward, terminated, truncated, info
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """
        Compute reward for the given achieved goal and desired goal.
        Only available in HER mode.
        
        Args:
            achieved_goal: The achieved goal.
            desired_goal: The desired goal.
            info: Additional information.
            
        Returns:
            float: Reward value.
        """
        if not self.use_her or not self.is_goal_env:
            raise ValueError("compute_reward can only be called in HER mode")
            
        return self.env.compute_reward(achieved_goal, desired_goal, info)
    
    def close(self):
        """Close the environment."""
        self.env.close()