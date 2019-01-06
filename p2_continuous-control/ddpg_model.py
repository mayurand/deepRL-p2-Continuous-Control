import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units = 256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        action = self.fc1(state)
        action = F.relu(action)
        action = self.fc2(action)
        return F.tanh(action) # Tanh activation function used for continuous action values
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

class Critic(nn.Module):
    """Critic (Action Value function) model"""
    
    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # The first layer takes in the state
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        
        # The second layer concatenates the actions with the output of previous layer
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        
        # Here the output is 1 because we want to get the Q values for input state and action
        self.fc4 = nn.Linear(fc3_units, 1) 
        self.reset_parameters()

    def forward(self, state, action):
        """Here the state and action both are taken as inputs for getting the Q value."""
        xs = F.leaky_relu(self.fc1(state))
        # Concatenate the action and the values from previous layer
        x = torch.cat((xs, action), dim=1) 
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x) # No activation as we want Q values directly
    
    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)



    
