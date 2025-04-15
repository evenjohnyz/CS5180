# utils/buffer.py
import numpy as np
import random

class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """
    def __init__(self, state_dim, action_dim, max_size=1e6):
        """
        Initialize replay buffer.
        
        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            max_size (int): Maximum size of the buffer.
        """
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        
        # Buffer storage
        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        
    def add(self, state, action, next_state, reward, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            next_state: Next state.
            reward: Reward received.
            done: Whether the episode has ended.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            tuple: Batch of (states, actions, next_states, rewards, not_dones)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )