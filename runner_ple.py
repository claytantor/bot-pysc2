import os, sys
import numpy as np
import time
import torch
import matplotlib
import matplotlib.pyplot as plt

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
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display



episode_scores = []

def show_screen(screen, name):
    plt.figure(2)
    plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
    plt.title(name)
    plt.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_scores(map_name=""):
    plt.figure(1)
    plt.clf()
    # scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training {}...'.format(map_name))
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(moving_average(episode_scores, 2))
    plt.plot(moving_average(episode_scores, 20))

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())                       

def get_screen(plenv, device):

    """
    obs = [x,y,3] RGB array
    """
    observation = plenv.getScreenRGB()

    # flip and rotate 90 deg
    flipped = np.rot90(np.flip(observation, axis=1), 1)  

    #Transpose it into torch order (CHW).
    screen = flipped.transpose((2, 0, 1))

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


def main(argv): 
    plt.ion()

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
    for i_episode in range(1000):  
        game.init()
        last_screen = get_screen(plenv, device)
        current_screen = get_screen(plenv, device)           
        state = current_screen - last_screen
        while not plenv.game_over():

            # returns the approiate action tensor and id
            t_action, action_num = agent.pickAction(state)       
            reward = plenv.act(action_num)

            if(reward>0.0):
                print("GOAL...")

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(plenv, device)

            if not game.game_over():
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory, needs to store the tensor action 
            # and the  call an optimization
            t_reward = torch.tensor([reward], device=device)
            agent.push(state, t_action, next_state, t_reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.update_network(i_episode)

            # if game.game_tick % np.random.randint(1,100) == 0 and state != None:
            #     show_screen(state, "state render")


        print("episode:{} score:{}".format( i_episode, game.getScore() ))       
        episode_scores.append(float(game.getScore()))
        plot_scores("MoveToBeacon")
        


if __name__ == "__main__":
    main(sys.argv[1:])
