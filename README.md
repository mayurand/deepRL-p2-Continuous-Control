# Udacity Deep Reinforcement Nanodegree Project 2: Continuous Control
This repository contains implementation of Continuous Control project as a part of Udacity's Deep Reinforcement Learning Nanodegree program.

In this project a double-jointed arm is trained to control its position for a moving target.
![Double Jointed Arms](images/trained_arms.gif)

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, Rand angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Getting Started

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Install Anaconda for Python3 from [here](https://www.anaconda.com/download).

2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name p2_drlnd python=3.6
	source activate p2_drlnd
	```
	- __Windows__: 
	```bash
	conda create --name p2_drlnd python=3.6 
	activate p2_drlnd
	```
	
3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/mayurand/deepRL-p2-Continuous-Control.git
cd deepRL-p2-Continuous-Control/python
pip install .
```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `p2_drlnd` environment.
```bash
python -m ipykernel install --user --name p2_drlnd --display-name "p2_drlnd"
```

