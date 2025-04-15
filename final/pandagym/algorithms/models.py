# algorithms/models.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network for DDPG.
    Maps states to deterministic actions.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            max_action (float): Maximum action value.
            hidden_dim (int): Hidden layer dimension.
        """
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state.
            
        Returns:
            torch.Tensor: Action scaled to [-max_action, max_action].
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.fc3(x))
        
        # Scale actions to the action space range
        return actions * self.max_action


class Critic(nn.Module):
    """
    Critic network for DDPG.
    Maps state-action pairs to Q-values.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            hidden_dim (int): Hidden layer dimension.
        """
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        """
        Forward pass through the network.
        
        Args:
            state: Input state.
            action: Input action.
            
        Returns:
            torch.Tensor: Q-value for the state-action pair.
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value