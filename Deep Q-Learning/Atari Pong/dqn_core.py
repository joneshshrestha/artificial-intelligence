"""
CSC 580 Winter 2026: dqn-core.py

Collection of core functions for Atari Pong.
"""
#---------------------------------------------------
# 1. Wrappers -- additional features for environment
#---------------------------------------------------
from re import A
import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import ale_py

class AtariPreprocess(gym.ObservationWrapper):
    """Observation wrapper, which converts raw Atari frames into
    84�84 grayscale images."""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs

class FrameStack(gym.Wrapper):
    """Observation wrapper, which stacks the last k frames together into 
    one observation.  Instead of seeing one static image, the agent sees 
    a short history of images, which lets it infer ball velocity, 
    direction of movement, and relative motion between ball and paddle.
    Without this, Pong is essentially unsolvable."""
    
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, 84, 84), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, terminated, truncated, info

class PongActionReducer(gym.ActionWrapper):
    """Observation wrapper, which reduce the action space from the original six 
    (NOOP, FIRE, UP, DOWN, RIGHTFIRE, LEFTFIRE) to four (NOOP, FIRE, UP, DOWN).
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self._action_map = [0, 1, 2, 3]  # NOOP, FIRE, UP, DOWN

    def action(self, act):
        return self._action_map[act]
		
#----------------------------------
# 2. Reply Buffer
#----------------------------------
import random
import pickle

class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy reinforcement learning (e.g., DQN).

    This buffer stores transitions of the form:
        (state, action, reward, next_state, done)

    Transitions are stored in a fixed-size circular buffer. When the buffer
    reaches its maximum capacity, newly added transitions overwrite the
    oldest ones.

    The replay buffer enables:
    - Breaking temporal correlations between consecutive samples
    - Reusing past experience for more data-efficient learning
    - Stabilizing training of value-based methods such as DQN

    Typical usage:
        buffer = ReplayBuffer(capacity=100000)
        buffer.push(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    Attributes
    ----------
    capacity : int
        Maximum number of transitions that can be stored in the buffer.
    buffer : list
        Internal storage for transitions.
    pos : int
        Current position in the circular buffer where the next transition
        will be written.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        ## Store transition in buffer (circular).
        ## Be sure not to extend beyond capacity as well as setting
        ## the self.pos correctly.
        
        # create transition buffer
        transition = (state, action, reward, next_state, done)
        # if buffer is not full, add transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        # if buffer is full, overwrite oldest transition at position self.pos
        else:
            self.buffer[self.pos] = transition
        
        # increment position
        self.pos = (self.pos + 1) % self.capacity


    def sample(self, batch_size):
        ## Sample a batch of transitions from buffer and return them
        ## in arrays (where each element such as states, actions, etc.
        ## are made into separate numpy arrays).
        
        # sample a batch of transitions from buffer
        buffer_batch = random.sample(self.buffer, batch_size)
        # create numpy arrays for each transition
        states, actions, rewards, next_states, dones = zip(*buffer_batch)
        # convert to numpy arrays and convert to float32 and int32
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
			
#--------------------------------------------------------------
# 3. DQN -- same deep learning network used in the Nature paper
#--------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) for Atari environments.

    This network approximates the action-value function Q(s, a) using a
    convolutional neural network that operates directly on raw pixel inputs.
    It follows the architecture introduced in the original DeepMind DQN paper
    for Atari games.

    Input
    -----
    x : torch.Tensor
        A batch of stacked Atari frames with shape:
            (batch_size, 4, 84, 84)
        Pixel values are expected to be in the range [0, 255] (uint8 or float).

    Output
    ------
    q_values : torch.Tensor
        A tensor of shape:
            (batch_size, n_actions)
        where each entry corresponds to the estimated Q-value for taking a
        particular action in the given state.

    Architecture
    ------------
    - Three convolutional layers with ReLU activations:
        * 32 filters, 8�8 kernel, stride 4
        * 64 filters, 4�4 kernel, stride 2
        * 64 filters, 3�3 kernel, stride 1
    - Fully connected layer with 512 hidden units
    - Output layer producing one Q-value per action

    Notes
    -----
    - Inputs are normalized by dividing pixel values by 255.0 inside
      the forward pass.
    - The network assumes 4-frame stacking to capture temporal information.
    - This model is typically used with experience replay and a target network
      for stable training.
    """
    
    def __init__(self, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        ## Normalize input, then and compute Q-values (output of the
        ## fully-connected (fc) block.  Return the q-values."""
        # normalize input x where x: (batch_size, 4, 84, 84)
        x = x.float() / 255.0
        # compute Q-values
        # pass through convolutional layers
        x = self.conv(x)
        # flatten x after convolutional layers
        x = torch.flatten(x, start_dim=1)
        # pass through fully-connected layers (Q-values)
        q_values = self.fc(x)
        # return Q-values
        return q_values



#--------------------------------------------------
# 4. make_env function -- incorporates the Wrappers
#--------------------------------------------------
def make_env(render_mode=None):
    env = gym.make("PongNoFrameskip-v4", render_mode=render_mode)
    env = PongActionReducer(env)
    env = AtariPreprocess(env)
    env = FrameStack(env, 4)
    return env
	