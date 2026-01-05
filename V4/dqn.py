import copy

import torch
from matplotlib.widgets import EllipseSelector
from torch.nn import *
from itertools import pairwise
from collections import deque
import random
import gymnasium as gym
from gymnasium.spaces.utils import *
from math import prod
import math

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.mps.is_available() else
    'cpu'
)


class DQN(Sequential):
    def __init__(self, hidden, input_shape, output_shape):
        super().__init__()
        n_inputs = prod(input_shape)
        n_outputs = prod(output_shape)
        dims = [n_inputs, *hidden, n_outputs]
        self.add_module("Flatten", Flatten())
        for idx, (i, o) in enumerate(pairwise(dims)):
            self.add_module(f'Linear_{idx}', Linear(i, o))
            self.add_module(f'RElU_{idx}', ReLU(inplace=True))
        self.add_module("Unflatten", Unflatten(1, output_shape))

class HyperParams:
    def __init__(self):
        self.mem_limit = 10000
        self.epsilon_start = 0.95
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.05
        self.episodes = 600

class ReinforcementProblem:
    def __init__(self, *, env: gym.Env, hyperparams: HyperParams, policy_net: Module, target_net: Module | None = None):
        super().__init__()
        
        self.env = env
        self.hyper = hyperparams
        self.memory = deque(maxlen=self.hyper.mem_limit)
        self.epsilon = 0.0
        self.observation = None
        self.info = None
        self.steps_done = 0
        self.policy_net = policy_net
        self.target_net = target_net if target_net is not None else policy_net
        
        self.reset()
        
    def reset(self):
        self.epsilon = self.hyper.epsilon_start
        obs, info = self.env.reset()
        self.observation = torch.from_numpy(obs).unsqueeze(0).to(device)
        self.steps_done = 0
    
    def _epsilon_greedy_action_select(self):
        epsilon = (self.hyper.epsilon_end + (self.hyper.epsilon_start - self.hyper.epsilon_end) *
                   math.exp(-1. * self.steps_done / self.hyper.epsilon_decay))
        self.steps_done += 1

        if random.uniform(0, 1) < epsilon:
            return torch.tensor([self.env.action_space.sample()], device=device, dtype=torch.dtype(self.env.action_space.dtype))
        else:
            with torch.no_grad():
                pred = self.policy_net(self.observation)

    
    def train(self, gif_export_path = None):

        for episode in range(self.hyper.episodes):
            self.reset()
            for step in range(self.hyper.)

