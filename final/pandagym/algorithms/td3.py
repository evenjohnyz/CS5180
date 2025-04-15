# algorithms/td3.py
import numpy as np
import torch
import torch.nn.functional as F
import copy
from algorithms.models import Actor, Critic
from algorithms.base_agent import BaseAgent

class TD3(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.
    Paper: https://arxiv.org/abs/1802.09477
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        hidden_dim=256,
        device="cpu"
    ):
        """
        Initialize TD3 agent.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            max_action (float): Maximum action value.
            discount (float): Discount factor for future rewards.
            tau (float): Target network update rate.
            policy_noise (float): Noise added to target policy.
            noise_clip (float): Range to clip target policy noise.
            policy_freq (int): Frequency of delayed policy updates.
            actor_lr (float): Learning rate for actor network.
            critic_lr (float): Learning rate for critic networks.
            hidden_dim (int): Hidden dimension for networks.
            device (str): Device to use for computation.
        """
        super(TD3, self).__init__(state_dim, action_dim, max_action)
        
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # TD3 uses twin critics to reduce overestimation bias
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Counter for delayed policy updates
        self.total_it = 0
        
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
        self.total_it += 1
        
        # Sample transitions from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)
        
        # Select next action according to target policy with noise
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            # Take the minimum of the two target Q-values to reduce overestimation
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.discount * target_Q)
        
        # Get current Q estimates
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        
        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_loss_history.append(critic_loss.item())
        
        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Delayed policy updates
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_loss_history.append(actor_loss.item())
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        actor_loss_value = actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss.item(),
            "q_value": current_Q1.mean().item()
        }
    
    def save(self, filename):
        """
        Save the agent's models.
        
        Args:
            filename (str): Path to save the models.
        """
        torch.save(self.critic1.state_dict(), f"{filename}_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{filename}_critic2.pth")
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
    
    def load(self, filename):
        """
        Load the agent's models.
        
        Args:
            filename (str): Path to load the models from.
        """
        self.critic1.load_state_dict(torch.load(f"{filename}_critic1.pth", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)
        
        self.critic2.load_state_dict(torch.load(f"{filename}_critic2.pth", map_location=self.device))
        self.critic2_target = copy.deepcopy(self.critic2)
        
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
        self.total_it += 1
        
        # Update critic networks
        with torch.no_grad():
            # Select next action according to target policy with noise
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)
            
            # Take minimum of both critics for target Q value
            target_Q1 = self.critic1_target(next_states, next_actions)
            target_Q2 = self.critic2_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (not_dones * self.discount * target_Q)
        
        # Get current Q estimates
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        
        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_loss_history.append(critic_loss.item())
        
        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Delayed policy updates
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_loss_history.append(actor_loss.item())
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "actor_loss": actor_loss if isinstance(actor_loss, (float, int)) else actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value": current_Q1.mean().item()
        }