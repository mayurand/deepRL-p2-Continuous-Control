import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    """
    This class implements a Neural Network as a function approximator for a policy.
    """
    
    def __init__(self, state_size=33, action_size=4, seed=0):
        
        # We need to initialize the nn.Module class within Policy(). 
        super(Policy, self).__init__() # The super initializes the nn.Module parentclass 
        h1_size=32
        h2_size=64
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.fc1 = nn.Linear(state_size, h1_size) # First layer
        self.fc2 = nn.Linear(h1_size, h2_size) # Second layer
        self.fc3 = nn.Linear(h2_size, action_size) # Output layer
        
    def forward(self, state):
        h1_state = F.relu(self.fc1(state))
        h2_state = F.relu(self.fc2(h1_state))
        action_probs = F.tanh(self.fc3(h2_state)) 
        return action_probs
    
    def act(self, state):
        # Convert the state (as a numpy array) into a torch tensor       
        state_in_torch = torch.from_numpy(state).float().unsqueeze(0) 
          
        # Pass the input state from the network and get action probs
        action_probs = self.forward(state_in_torch)
        return action_probs