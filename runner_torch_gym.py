import random
import sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

from envs.sc2gym import SC2GymEnv
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# from pysc2.lib import actions, features, units

from agents.terran_move_beacon import TerranMoveToBeaconAgent

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


def main(argv):   
    print("starting sc2 bot app.")
    load_dotenv(find_dotenv())

    replaysDir = os.path.join(Path(__file__).parent.absolute(),'replays')

    env = SC2GymEnv("MoveToBeacon", realtime=True, visualize=False, replay_dir=replaysDir)

    agent = TerranMoveToBeaconAgent()

    agent.setup(env.observation_spec, env.action_spec)

    # training loop
    score = 0

    # Requires 100 episodes for evaluation
    for i_episodes in range(100):
        # Reset the environment for the current episode
        obs = env.reset()
        agent.reset()

        # Set up a loop to perform 1000 steps
        for t in range(100):
            step_actions = [agent.step(obs)]                 
            obs, reward, done, _ = env.step(step_actions)
            score += reward
            if done:
                plt.figure()
                plt.imshow(get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy(),
                        interpolation='none')
                plt.title('Example extracted screen')
                plt.show()
                print("Episode finished after {} timepsteps. score: {}".format(t+1, score))
                break


if __name__ == "__main__":
    main(sys.argv[1:])
