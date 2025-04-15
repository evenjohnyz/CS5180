# algorithms/sac.py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from algorithms.sac_models import GaussianPolicy, SoftQNetwork
from algorithms.base_agent import BaseAgent

class SAC(BaseAgent):
    """
    Soft Actor-Critic (SAC) algorithm.
    Paper: https://arxiv.org/abs/1801.01290
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        hidden_dim=256,
        device="cpu"
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            max_action (float): Maximum action value.
            discount (float): Discount factor for future rewards.
            tau (float): Target network update rate.
            alpha (float): Initial entropy coefficient value.
            automatic_entropy_tuning (bool): Whether to automatically tune alpha.
            actor_lr (float): Learning rate for actor network.
            critic_lr (float): Learning rate for critic networks.
            alpha_lr (float): Learning rate for alpha.
            hidden_dim (int): Hidden dimension for networks.
            device (str): Device to use for computation.
        """
        super(SAC, self).__init__(state_dim, action_dim, max_action)
        
        self.discount = discount
        self.tau = tau
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Initialize critic networks
        self.critic1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        
        self.critic2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Initialize policy network
        self.policy = GaussianPolicy(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        
        # Entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Statistics
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_loss_history = []
        
    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state.
            evaluate (bool): Whether to use deterministic action (for evaluation).
            
        Returns:
            np.ndarray: Selected action.
        """
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        # Get action from policy network
        with torch.no_grad():
            if evaluate:
                action = self.policy.get_action(state).cpu().data.numpy().flatten()
            else:
                action, _, _, _, _ = self.policy.sample(state)
                action = action.cpu().data.numpy().flatten()
        
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
        
        # Update critic networks
        with torch.no_grad():
            # Sample next action from policy and compute target Q-value
            next_action, next_log_prob, _, _, _ = self.policy.sample(next_state)
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * self.discount * target_Q
        
        # Compute current Q-values
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
        
        # Update actor network
        actions_new, log_prob, _, _, _ = self.policy.sample(state)
        q1_new = self.critic1(state, actions_new)
        q2_new = self.critic2(state, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        # Compute actor loss
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_loss_history.append(actor_loss.item())
        
        # Optimize the actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()
        
        # Optionally update alpha (entropy coefficient)
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_loss_history.append(alpha_loss.item())
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Update target networks with soft update
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
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
        torch.save(self.policy.state_dict(), f"{filename}_policy.pth")
        
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, f"{filename}_log_alpha.pth")
    
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
        
        self.policy.load_state_dict(torch.load(f"{filename}_policy.pth", map_location=self.device))
        
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(f"{filename}_log_alpha.pth", map_location=self.device)
            self.alpha = self.log_alpha.exp().item()

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
        # Update critic networks
        with torch.no_grad():
            # Sample next action from policy and compute target Q-value
            next_actions, next_log_probs, _, _, _ = self.policy.sample(next_states)
            target_Q1 = self.critic1_target(next_states, next_actions)
            target_Q2 = self.critic2_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_probs
            target_Q = rewards + not_dones * self.discount * target_Q
        
        # Compute current Q-values
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
        
        # Update actor network
        actions_new, log_probs, _, _, _ = self.policy.sample(states)
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        # Compute actor loss
        actor_loss = (self.alpha * log_probs - q_new).mean()
        self.actor_loss_history.append(actor_loss.item())
        
        # Optimize the actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()
        
        # Optionally update alpha (entropy coefficient)
        alpha_loss = 0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_loss_history.append(alpha_loss.item())
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Update target networks with soft update
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item() if self.automatic_entropy_tuning else 0,
            "alpha": self.alpha,
            "q_value": current_Q1.mean().item()
        }