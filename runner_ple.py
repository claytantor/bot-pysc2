import numpy as np
import time
import torch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from ple import PLE
from games.m2b import MoveToBeacon
from agents.m2b.drl import DRLAgent

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(plenv, device):

    """
    obs = [x,y,3] RGB array
    """
    observation = plenv.getScreenRGB()
    screen = observation.transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0).to(device)

class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]


game = MoveToBeacon()

plenv = PLE(game, fps=30, display_screen=True)
init_screen = get_screen(plenv, device)
agent = DRLAgent(actions=plenv.getActionSet(), init_screen=init_screen)
plenv.init()

# lets do a random number of NOOP's
for i in range(np.random.randint(0, 4)):
    reward = plenv.act(plenv.NOOP)

reward = 0.0

# number of elpisodes
for i in range(1000):  
    while not plenv.game_over():

        last_screen = get_screen(plenv,device)
        current_screen = get_screen(plenv,device)
        state = current_screen - last_screen

        # observation = plenv.getScreenRGB()
        action = agent.pickAction(state)
        

        # use the tensor number to 
        ## index to the action
        action_index = list(action.data.cpu().numpy()[0])[0]
        action_num = plenv.getActionSet()[action_index]
        # print(action_num)

        reward = plenv.act(action_num)
        if(reward>0.0):
            print("GOAL...")

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(plenv,device)
        if not game.game_over():
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        t_reward = torch.tensor([reward], device=device)
        agent.push(state, action, next_state, t_reward)

        # Move to the next state
        state = next_state

        # # Perform one step of the optimization (on the target network)
        agent.optimize_model()

        # ===


    print("episode:{} score:{}".format( i, game.getScore() ))
    plenv.reset_game()
    plenv.init()
    plenv.act(plenv.NOOP)
    
    time.sleep(1)

