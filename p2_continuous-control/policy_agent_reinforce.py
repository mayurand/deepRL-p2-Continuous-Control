import numpy as np
import random
from collections import namedtuple, deque

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
        action_probs = self.forward(state_in_torch).cpu()
        return action_probs

class Policy_REINFORCE(nn.Module):
    """
    This class implements a Neural Network as a function approximator for a policy.
    """
    
    def __init__(self, state_size=33, action_size=4, seed=0):
        
        # We need to initialize the nn.Module class within Policy(). 
        super(Policy_REINFORCE, self).__init__() # The super initializes the nn.Module parentclass 
        h1_size=32
        h2_size=64
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.fc1 = nn.Linear(state_size, h1_size) # First layer
        self.fc2 = nn.Linear(h1_size, h2_size) # Second layer
        
        # The output layer gives the mean and variance values. Thus action_size*2 are the output values.
        self.fc3 = nn.Linear(h2_size, action_size*2) # Output layer 
        
    def forward(self, state):
        
        h1_state = F.relu(self.fc1(state))
        h2_state = F.relu(self.fc2(h1_state))
        action_probs = F.softmax(self.fc3(h2_state), dim=1)
        return action_probs
    
    def act(self, state):
        # Convert the state (as a numpy array) into a torch tensor       
        state_in_torch = torch.from_numpy(state).float().unsqueeze(0).to(device)
          
        # Pass the input state from the network and get action probs
        action_probs = self.forward(state_in_torch).cpu()
        
        return action_probs
    
    
    
def collect_random_trajectory(env, policy, tmax=500):
    pass


def collect_trajectory_REINFORCE(env, policy, tmax=500):
    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    action_vals_list=[]
    actions = []
    
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    
    # For the number of time steps
    for t in range(tmax):
    
        action_vals = policy.act(state).detach().numpy()   # select an action
        
        action = []
        # Draw the actions from normal distribution
        for i in range(0,len(action_vals[0]),2):
            mean = action_vals[0][i]
            variance = action_vals[0][i+1]
            a = np.random.normal(mean,variance) 
            action.append(a)
                  
        action = np.clip(action, -1, 1)              # all actions between -1 and 1
        env_info = env.step(action)[brain_name]       # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
    
        # store the result
        state_list.append(state)
        reward_list.append(reward)
        action_vals_list.append(action_vals)
        actions.append(action)
        
        state = next_state                             # roll over the state to next time step
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if done:
            break

    action_vals_list = torch.Tensor(action_vals_list).to(device)
            
    # return action vals, states, rewards
    return action_vals_list, actions , state_list, reward_list



def collect_trajectory(env, policy, tmax=500):
    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    action_vals_list=[]
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    
    # For the number of time steps
    for t in range(tmax):
    
        action_val = policy.act(state)   # select an action
        #actions = np.clip(action_val.detach().numpy(), -1, 1)              # all actions between -1 and 1
        env_info = env.step(action_val.detach().numpy())[brain_name]       # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
    
        # store the result
        state_list.append(state)
        reward_list.append(reward)
        action_vals_list.append(action_val)
        
        state = next_state                             # roll over the state to next time step
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if done:
            break

    # return action vals, states, rewards
    return action_vals_list, state_list, reward_list



