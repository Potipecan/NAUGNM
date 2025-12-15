import torch
from matplotlib.widgets import EllipseSelector
from torch.nn import *
from itertools import pairwise
from collections import deque
import random
import gymnasium as gym
from gymnasium.spaces.utils import *
from math import prod

class HyperParams:
    def __init__(self):
        self.mem_limit = 10000
        self.epsilon_start = 0.95
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.05
        self.hidden_layers = [128]
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.mps.is_available() else
            'cpu'
        )

class DQN:
    def __init__(self, env: gym.Env, hyperparams: HyperParams):
        super().__init__()
        
        self.env = env
        self.hyper = hyperparams
        self.action_nn = DQN.make_nn(self.hyper.hidden_layers, self.env.observation_space.shape, self.env.action_space.shape).to(self.hyper.device)
        self.memory = deque(maxlen=self.hyper.mem_limit)
        self.epsilon = 0.0
        self.observation = None
        self.info = None
        
        self.reset()
        
    def reset(self):
        self.epsilon = self.hyper.epsilon_start
        obs, info = self.env.reset()
        self.observation = torch.from_numpy(obs).unsqueeze(0).to(self.hyper.device)

    @staticmethod
    def make_nn(hidden, input_shape, output_shape):
        n_inputs = prod(input_shape)
        n_outputs = prod(output_shape)
        dims = [n_inputs, *hidden, n_outputs]
        nn = Sequential()
        nn.add_module("Flatten", Flatten())
        for idx, (i, o) in enumerate(pairwise(dims)):
            nn.add_module(f'Linear_{idx}', Linear(i, o))
            nn.add_module(f'RElU_{idx}', ReLU(inplace=True))
        nn.add_module("Unflatten", Unflatten(1, output_shape))
        return nn
        
    
    def select_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            pred = self.action_nn(self.observation)
            selection = torch.argmax(pred, dim)
    
    
            