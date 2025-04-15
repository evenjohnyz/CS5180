# algorithms/ddpg.py
import numpy as np
import torch
import torch.nn.functional as F
import copy
from algorithms.models import Actor, Critic
from algorithms.base_agent import BaseAgent

class DDPG(BaseAgent):
    """
    Deep Deterministic Policy Gradient (DDPG) algorithm.
    Paper: https://arxiv.org/abs/1509.02971
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        hidden_dim=256,
        device="cpu"
    ):
        """
        Initialize DDPG agent.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            max_action (float): Maximum action value.
            discount (float): Discount factor for future rewards.
            tau (float): Target network update rate.
            actor_lr (float): Learning rate for actor.
            critic_lr (float): Learning rate for critic.
            hidden_dim (int): Hidden dimension for networks.
            device (str): Device to use for computation.
        """
        super(DDPG, self).__init__(state_dim, action_dim, max_action)
        
        self.discount = discount
        self.tau = tau
        self.device = device
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Statistics
        self.actor_loss_history = []
        self.critic_loss_history = []
        
    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state.
            evaluate (bool): Whether to add noise for exploration.
            
        Returns:
            np.ndarray: Selected action.
        """
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        # Get action from actor network
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        return action
    
    def train(self, replay_buffer, batch_size=256):
        """
        Update actor and critic networks using experience replay.
        
        Args:
            replay_buffer: Replay buffer containing experiences.
            batch_size (int): Batch size for training.
            
        Returns:
            dict: Training metrics.
        """
        # Sample transitions from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)
        
        # Update critic
        # Get target Q-value
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (not_done * self.discount * target_Q)
        
        # Get current Q-value
        current_Q = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_loss_history.append(critic_loss.item())
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_loss_history.append(actor_loss.item())
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value": current_Q.mean().item()
        }
    
    def save(self, filename):
        """
        Save the agent's models.
        
        Args:
            filename (str): Path to save the models.
        """
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
    
    def load(self, filename):
        """
        Load the agent's models.
        
        Args:
            filename (str): Path to load the models from.
        """
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

    def train_on_batch(self, states, actions, next_states, rewards, not_dones):
        """
        Train on a batch of transitions (for HER).
        
        Args:
            states: Batch of states (including goals).
            actions: Batch of actions.
            next_states: Batch of next states (including goals).
            rewards: Batch of rewards.
            not_dones: Batch of not_done flags.
            
        Returns:
            dict: Training metrics.
        """
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (not_dones * self.discount * target_Q)
        
        # Get current Q
        current_Q = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_loss_history.append(critic_loss.item())
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_loss_history.append(actor_loss.item())
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value": current_Q.mean().item()
        }