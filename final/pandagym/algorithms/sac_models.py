# algorithms/sac_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class GaussianPolicy(nn.Module):
    """
    Gaussian Policy for SAC.
    Maps states to a Gaussian distribution over actions.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log_std layers for the Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state.
            
        Returns:
            mean, log_std: Parameters of the Gaussian distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample an action from the Gaussian policy.
        
        Args:
            state: Input state.
            
        Returns:
            action, log_prob, z, mean, log_std: Sampled action, its log probability, 
                                                the pre-tanh sample, mean, and log_std.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from the Gaussian distribution
        normal = Normal(mean, std)
        z = normal.rsample()  # reparameterization trick
        
        # Squash the sample using tanh for bounded actions
        action = torch.tanh(z)
        
        # Calculate log_prob of the action
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Scale action to the action space range
        action = action * self.max_action
        
        return action, log_prob, z, mean, log_std
    
    def get_action(self, state):
        """
        Get a deterministic action from the policy for evaluation.
        
        Args:
            state: Input state.
            
        Returns:
            action: Deterministic action.
        """
        mean, _ = self.forward(state)
        action = torch.tanh(mean) * self.max_action
        return action

class SoftQNetwork(nn.Module):
    """
    Soft Q-Network for SAC.
    Maps state-action pairs to Q-values.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        
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
        x = self.fc3(x)
        
        return x