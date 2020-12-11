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

        # torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.001, alpha=0.99, #eps=EPS_START, weight_decay=EPS_DECAY, momentum=0.9, centered=False)

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)


         # optimizer
        # self.momentum = momentum
        if OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        elif OPTIMIZER == 'Nesterov':
            self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)


        self.memory = ReplayMemory(10000)

        self.steps_done = 0


    def push(self, state, action, next_state, reward):

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


 



class DRLCNNAgent():
    '''
    Environment details:
         Observations:
                There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger
                (including the case when the passenger is the taxi), and 4 destination locations.
         Actions:
            There are 6 discrete deterministic actions:
            - 0: move south
            - 1: move north
            - 2: move east
            - 3: move west
            - 4: pickup passenger
            - 5: dropoff passenger
        Rewards:
            There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger.
            There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
        Rendering:
            - blue: passenger
            - magenta: destination
            - yellow: empty taxi
            - green: full taxi
            - other letters: locations
    '''
    def __init__(
        self,
        env,
        name='',
        n_hidden=150,
        optimizer='RMSprop',
        momentum=0.9,
        loss='MSE',
        exploration=None,
        use_l1_regularizer=False,
        l1_lambda=1.0,
        replay_buffer_size=100000,
        gamma=0.99,
        learning_rate=0.0003,
        steps_to_start_learn=50000,
        target_update_freq=10000,
        obs_represent='one-hot',
        clip_grads=False,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.5
        ):


        assert type(env.observation_space) == gym.spaces.Discrete
        assert type(env.action_space) == gym.spaces.Discrete
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Hyper-params
        self.name = name
        self.gamma = gamma
        self.n_hidden = n_hidden
        self.n_actions = env.action_space.n
        self.n_obs = env.observation_space.n
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, 1)
        self.steps_to_start_learn = steps_to_start_learn
        self.target_update_freq = target_update_freq
        self.clip_grads=clip_grads
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_l1_regularizer = use_l1_regularizer
        self.l1_lambda = l1_lambda
        self.obs_represent = obs_represent
        if (self.obs_represent == 'one-hot'):
            print("Using One-Hot-Vector representation for states")
            self.one_hot_generator = OneHotGenerator(self.n_obs)
        elif self.obs_represent == 'state-int':
            self.n_obs = 1
            print("Using State-Integer representation for states")
        elif self.obs_represent == 'location-one-hot':
            print("Using Location-One-Hots representation for states")
            self.n_obs = 19 # 5 bits for row, col, passenger location and 4 for destIdx
        else:
            self.n_obs = len(list(env.env.decode(0))) # get state of the game as a tuple
            print("Using Locations-Tuple representation for states")
        if loss == 'SmoothL1':
            self.loss_criterion = nn.SmoothL1Loss()
        else:
            self.loss_criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        if exploration is None:
            self.explore_schedule = ExponentialSchedule()
        else:
            self.explore_schedule = exploration

        self.epsilon = self.explore_schedule.value(0)

        # Initialize DQN's and optimizer
        self.Q_train = DQN_DNN(
            self.n_obs,
            self.n_hidden,
            self.n_actions,
            self.use_batch_norm,
            self.use_dropout,
            self.dropout_rate).to(self.device)

        self.Q_target = DQN_DNN(
            self.n_obs,
            self.n_hidden,
            self.n_actions,
            self.use_batch_norm,
            self.use_dropout,
            self.dropout_rate).to(self.device)

        # set modes
        self.Q_train.train()
        self.Q_target.eval()

        # optimizer
        self.momentum = momentum
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.Q_train.parameters(), lr=self.learning_rate)
        elif optimizer == 'Nesterov':
            self.optimizer = torch.optim.SGD(self.Q_train.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.RMSprop(self.Q_train.parameters(), lr=self.learning_rate)

        # statistics
        self.steps_count = 0
        self.episodes_seen = 0
        self.num_param_updates = 0

        # load checkpoint if it exists
        self.load_agent_state()

        # init target network with the same weights
        self.Q_target.load_state_dict(self.Q_train.state_dict())

        print("Created Agent for Taxi-v2")

    def select_greedy_action(self, obs):
        '''
        This method picks an action to perform according to an epsilon-greedy policy.
        Parameters:
            obs: current state or observation from the environment (int or tuple)
                for One-Hot the state is a number (out of 500)
                else the state is a tuple (taxiRow, taxiCol, passLoc, destIdx)
        Returns:
            action (int)
        '''
        self.epsilon = self.explore_schedule.value(self.episodes_seen)
        threshold = self.epsilon
        rand_num = random.random()
        if (rand_num > threshold):
            # Pick according to current Q-values
            if self.obs_represent == 'one-hot' or self.obs_represent == 'location-one-hot':
                if type(obs) is int:
                    obs = self.one_hot_generator.to_one_hot(obs)
            elif self.obs_represent == 'state-int':
                obs = obs / 499.0
            else:
                assert len(obs) == 4
                obs = obs / 4.0 # normalize
            # make sure the target network is in eval mode (no gradient calculation)
            obs = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor).to(self.device)
            self.Q_target.eval()
            with torch.no_grad():
                q_actions = self.Q_target(obs)
                action = torch.argmax(q_actions).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn(self, batch_size):
        '''
        This method performs a training step for the agent.
        Parmeters:
            batch_size: number of samples to perform the training step on (int)
        '''
        if (self.steps_count > self.steps_to_start_learn and
            self.replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(batch_size)
            obs_batch = torch.from_numpy(obs_batch).type(torch.FloatTensor).to(self.device)
            act_batch = torch.from_numpy(act_batch).long().to(self.device)
            rew_batch = torch.from_numpy(rew_batch).to(self.device)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(torch.FloatTensor).to(self.device)
            if self.obs_represent == 'locations':
                obs_batch = obs_batch / 4.0 # normalize
                next_obs_batch = next_obs_batch / 4.0 # noramalize
            elif self.obs_represent == 'state-int':
                obs_batch = obs_batch / 499.0 # normalize
                next_obs_batch = next_obs_batch / 499.0 # noramalize
            not_done_mask = torch.from_numpy(1 - done_mask).to(self.device)
            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = self.Q_train(obs_batch.type(torch.FloatTensor).to(self.device)).gather(1, act_batch.unsqueeze(1))
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = self.Q_target(next_obs_batch.type(torch.FloatTensor).to(self.device)).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = rew_batch + (self.gamma * next_Q_values)
            # loss
            loss = self.loss_criterion(current_Q_values, target_Q_values.unsqueeze(1))
            # add regularozation
            if self.use_l1_regularizer:
                lam = torch.tensor(self.l1_lambda)
                l1_reg = torch.tensor(0.)
                for param in self.Q_train.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += lam * l1_reg
            # optimize model
            self.optimizer.zero_grad()
            loss.backward()
            if (self.clip_grads):
                for param in self.Q_train.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.num_param_updates += 1
            # copy weights to target network and save network state
            if self.num_param_updates % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q_train.state_dict())
                # save network state
                self.save_agent_state()

    def predict_action(self, obs):
        '''
        Predict action for inference or playing.
        Parameters:
            obs: a state observation from the environment (int/tuple/np.array)
        Returns:
            action: action for which the Q-value of the current observation is the highest (int)
        '''
        with torch.no_grad():
            obs = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor).to(self.device)
            q_actions = self.Q_target(obs)
            action = torch.argmax(q_actions).item()
        return action

    def save_agent_state(self):
        '''
        This function saves the current state of the DQN (the weights) to a local file.
        '''
        filename = "taxi_agent_" + self.name + ".pth"
        dir_name = './taxi_agent_ckpt'
        full_path = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save({
            'model_state_dict': self.Q_train.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_count': self.steps_count,
            'episodes_seen': self.episodes_seen,
            'epsilon': self.epsilon,
            'num_param_updates': self.num_param_updates
            }, full_path)
        print("Saved Taxi Agent checkpoint @ ", full_path)

    def load_agent_state(self, path=None, copy_to_target_network=False, load_optimizer=True):
        '''
        This function loads an agent checkpoint.
        Parameters:
            path: path to a checkpoint, e.g `/path/to/dir/ckpt.pth` (str)
            copy_to_target_network: whether or not to copy the loaded training
                DQN parameters to the target DQN, for manual loading (bool)
            load_optimizer: whether or not to restore the optimizer state
        '''
        if path is None:
            filename = "taxi_agent_" + self.name + ".pth"
            dir_name = './taxi_agent_ckpt'
            full_path = os.path.join(dir_name, filename)
        else:
            full_path = path
        exists = os.path.isfile(full_path)
        if exists:
            if not torch.cuda.is_available():
                checkpoint = torch.load(full_path, map_location='cpu')
            else:
                checkpoint = torch.load(full_path)
            self.Q_train.load_state_dict(checkpoint['model_state_dict'])
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_count = checkpoint['steps_count']
            self.episodes_seen = checkpoint['episodes_seen']
            self.epsilon = checkpoint['epsilon']
            self.num_param_update = checkpoint['num_param_updates']
            print("Checkpoint loaded successfully from ", full_path)
            # for manual loading a checkpoint
            if copy_to_target_network:
                self.Q_target.load_state_dict(self.Q_train.state_dict())
        else:
            print("No checkpoint found...")

def preprocess_frame(env, mode='atari', render=False):
    '''
    This function preprocess the current frame of the environment.
    Prameters:
        env: the environment (gym.Env)
        mode: processing mode to use (str)
            options: 'atari' - 1 channel, 'control' - 3 channels
        render: whetheState-Integerr or not to render the screen, which opens a window (bool)
            * in ClassicControl problems, even when setting render mode to 'rgb_array',
                a window is opened. Setting this to False will close this window each time.
            * Performance is better when set to True, less overhead.
    Returns:
        frame: the processed (reshaped, scaled, adjusted) frame (np.array, np.uint8)
    '''
    screen = env.render(mode='rgb_array')
    if not render:
        env.close() # on Windows, must close the opened window
    if mode =='atari':
        screen = np.reshape(screen, [500, 500, 3]).astype(np.float32)
        screen = screen[:, :, 0] * 0.299 + screen[:, :, 1] * 0.587 + screen[:, :, 2] * 0.114 # dimension reduction, contrast
        screen = Image.fromarray(screen)
        resized_screen = screen.resize((84, 84), Image.BILINEAR)
        resized_screen = np.array(resized_screen)
        x_t = np.reshape(resized_screen, [84, 84, 1])
    else:
        screen = np.reshape(screen, [500, 500, 3]).astype(np.uint8)
        screen = Image.fromarray(screen)
        resized_screen = screen.resize((84, 84), Image.BILINEAR)
        resized_screen = np.array(resized_screen)
        x_t = np.reshape(resized_screen, [84, 84, 3])
    return x_t.astype(np.uint8)

