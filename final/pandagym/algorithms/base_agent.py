# algorithms/base_agent.py
import numpy as np
import torch
import torch.nn as nn

class BaseAgent:
    """
    Base class for all RL agents.
    """
    def __init__(self, state_dim, action_dim, max_action):
        """
        Initialize the agent.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            max_action (float): Maximum action value.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
    def select_action(self, state):
        """
        Select an action given the current state.
        
        Args:
            state: Current state.
            
        Returns:
            np.ndarray: Selected action.
        """
        raise NotImplementedError
        
    def train(self, replay_buffer, batch_size=256):
        """
        Update the agent's networks using a batch of experiences.
        
        Args:
            replay_buffer: Experience replay buffer.
            batch_size (int): Size of the batch to sample.
            
        Returns:
            dict: Training metrics.
        """
        raise NotImplementedError
        
    def save(self, filename):
        """
        Save the agent's models.
        
        Args:
            filename (str): Path to save the models.
        """
        raise NotImplementedError
        
    def load(self, filename):
        """
        Load the agent's models.
        
        Args:
            filename (str): Path to load the models from.
        """
        raise NotImplementedError