import sys
import os
import logging
import random
import math

from dotenv import load_dotenv, find_dotenv
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

#         # convert to tensor
#         Exception has occurred: RuntimeError
# Expected 4-dimensional input for 4-dimensional weight [16, 3, 5, 5], but got 2-dimensional input of size [64, 64] instead

        x = torch.from_numpy(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



class TerranMoveToBeaconAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranMoveToBeaconAgent, self).__init__()
        # self.attack_coordinates = None    

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(TerranMoveToBeaconAgent, self).step(obs)

        self.last_screen = obs.observation['feature_minimap'][0]
        self.current_screen = obs.observation['feature_minimap'][0]
        
        self.state = self.current_screen - self.last_screen
        # reward = torch.tensor([obs.reward], device=device)

        self.first(obs)
        actions_list = []
        actions_list.append(self.move_marine(obs))
        actions_list.append(self.choose_action_drl(obs, self.state))
        

        #filter out empties
        active_actions = list(filter(lambda x: x != None, actions_list))

        # return the chosed action
        if len(active_actions)==0:
            return actions.FUNCTIONS.no_op()
        else:
            return active_actions[0]
    
    def setup(self, obs_spec, action_spec):

        super(TerranMoveToBeaconAgent, self).setup(obs_spec, action_spec)

        # Get number of actions from gym action space
        self.n_actions = len(action_spec)
        feature_minimap_shape = obs_spec[0]['feature_minimap']

        self.policy_net = DQN(feature_minimap_shape[1], feature_minimap_shape[2], self.n_actions).to(device)
        self.target_net = DQN(feature_minimap_shape[1], feature_minimap_shape[2], self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

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

    def push(self, action, current_screen, reward, done):

        # Observe new state
        self.last_screen = self.current_screen
        self.current_screen = current_screen
        if not done:
            next_state = current_screen - self.last_screen
        else:
            next_state = None

        self.memory.push(self.state, action, next_state, reward)
        # Move to the next state
        self.state = next_state
        self.optimize_model()

    def first(self, obs):
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()
            logger.info("first x_mean:{} y_mean:{}".format(xmean, ymean))

    def move_marine(self, obs):

        marines = self.get_units_by_type(obs, units.Terran.Marine)
        if len(marines) >= 0:
            if self.unit_type_is_selected(obs, units.Terran.Marine):

                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    # should be shape
                    x_val = random.randint(0, 63)
                    y_val = random.randint(0, 63)
       
                    return actions.FUNCTIONS.Attack_minimap("now",
                            (x_val, y_val))
                
            else:
                marine = random.choice(marines)          
                return actions.FUNCTIONS.select_point("select_all_type", (marine.x, marine.y))



        return None


    def choose_action_drl(self, obs, state):


        # state = torch.from_numpy(state.values)
     
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # return self.policy_net(state).max(1)[1].view(1, 1)
                value = self.policy_net(state).max(1)[1].view(1, 1)
                print("choose_action_drl", value)
        else:
            # return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
            value = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
            print("choose_action_drl", value)

        return None
