import sys, os
import gym
import math
import random 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
OPTIMIZER = 'Nesterov'

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN_CNN(nn.Module):
    def __init__(self, n_actions, in_channels=4, use_batch_norm=False, use_dropout=False, dropout_rate=0.5, mode='control'):
        '''
        Parmaeters:
            in_channels: number of input channels (int)
                e.g. The number of most recent frames stacked together as describe in the paper: 
                    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
            n_actions: number of possible actions in the environment, the number of Q-values (int)
            use_batch_norm: use batch normalization agter the first layer (bool)
            use_dropout: add dropout regulariztion (bool)
            dropout_rate: probability of a neuron to be dropped during training (float)
            mode: the architecture is determined according to the task at hand (str)
                'control' - Acrobot-v1, stacked frames / difference of frames as state
                'atari' - Atari games, stacked frames based
        '''
        super(DQN_CNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.mode = mode
        # Conv layers
        # Dimensions formula:
        # Hout = floor[(Hin + 2*padding[0] - dialation[0] * (kernel_size[0] -
        # 1) - 1)/(stride[0]) + 1]
        # Wout = floor[(Win + 2*padding[1] - dialation[1] * (kernel_size[1] -
        # 1) - 1)/(stride[1]) + 1]
        if self.mode == 'control':
            if self.use_batch_norm:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)),
                    ('bn1', nn.BatchNorm2d(16)),  
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(16, 32, kernel_size=5, stride=2)),
                    ('bn2', nn.BatchNorm2d(32)),  
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(32, 32, kernel_size=5, stride=2)),
                    ('bn3', nn.BatchNorm2d(32)),  
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                self.fc1 = nn.Linear(32 * 7 * 7, n_actions)
            else:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)), 
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(16, 32, kernel_size=5, stride=2)), 
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(32, 32, kernel_size=5, stride=2)),
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                self.fc1 = nn.Linear(32 * 7 * 7, n_actions)
        else: # mode == 'atari'
            if self.use_batch_norm:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
                    ('bn1', nn.BatchNorm2d(32)),  
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                    ('bn2', nn.BatchNorm2d(64)),  
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                    ('bn3', nn.BatchNorm2d(64)),  
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                fc1_layers = OrderedDict([('fc1', nn.Linear(64 * 7 * 7, 512)),
                                          ('bn4', nn.BatchNorm1d(512)),
                                          ('relu4', nn.ReLU())])
                self.fc1 = nn.Sequential(fc1_layers)
                self.fc2 = nn.Linear(512, n_actions)
            else:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2)), 
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                fc1_layers = OrderedDict([('fc1', nn.Linear(64 * 7 * 7, 512)),
                    ('relu4', nn.ReLU())])
                self.fc1 = nn.Sequential(fc1_layers)
                self.fc2 = nn.Linear(512, n_actions)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        '''
        Forward pass.
        '''
        if self.mode == 'control':
            if self.use_dropout:
                x = self.conv1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                return x
            else:
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                return x
        else:
            # mode == 'atari'
            if self.use_dropout:
                x = self.conv1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                x = self.fc2(x)
                return x
            else:
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                x = self.fc2(x)
                return x


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

"""
If the observations are images we use CNNs.
"""
class QNetworkCNN(nn.Module):
    def __init__(self, action_dim):
        super(QNetworkCNN, self).__init__()

        self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(8960, 512)
        self.fc_2 = nn.Linear(512, action_dim)

    def forward(self, inp):
        inp = inp.view((1, 3, 210, 160))
        x1 = F.relu(self.conv_1(inp))
        x1 = F.relu(self.conv_2(x1))
        x1 = F.relu(self.conv_3(x1))
        x1 = torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = self.fc_2(x1)

        return x1

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DRLAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions, init_screen):
        self.actions = actions
        self.learning_rate = 0.01
        self.momentum = 0.9


        # Get number of actions from gym action space
        self.n_actions = len(self.actions)

        # set local vars
        # _, _, screen_height, screen_width = self.get_screen(screen)
        _, _, screen_height, screen_width = init_screen.shape
        self.screen_height = screen_height
        self.screen_width = screen_width

        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        if OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        elif OPTIMIZER == 'Nesterov':
            self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)

        self.memory = ReplayMemory(10000)

        self.steps_done = 0


    def push(self, state, action, next_state, reward):

        print("lr",)

        # send step to memory
        self.memory.push(state, action, next_state, reward)

        # # Perform one step of the optimization (on the target network)
        self.optimize_model()


    def update_network(self, i_episode):
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def pickAction(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(state).max(1)[1].view(1, 1)
                action_index = list(action[0])[0]
                return action, self.actions[action_index]
              
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
            action_index = list(action[0])[0]
            return action, self.actions[action_index]



    def optimize_model(self):

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


 

