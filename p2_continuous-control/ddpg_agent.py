import numpy as np
import random
import copy

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ddpg_model import Actor, Critic


class Agent():
    """Interacts with and learns from the environment"""
    
    def __init__(self, state_size, action_size, random_seed):
        """
        Initialize the agent object
        
        Params:
        ======
            state_size (int): dimension of each state 
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        
        
        # Critic Network (w/ Target Network)
        
        
    def step():
        pass
    
    def act():
        pass
    
    
    def reset():
        pass
    
    
    def learn():
        pass
    
    def soft_update():
        pass
    
        
        
        