import sys
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

from agents.pandas_qtable import QTableAgent

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import numpy as np

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_scores = []

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
    plt.plot(moving_average(episode_scores, 100))

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())      



def main(unused_argv):
    print("starting sc2 bot app.")

    map_name="MoveToBeacon"

    load_dotenv(find_dotenv())
    
    agent = QTableAgent()

    replaysDir = os.path.join(Path(__file__).parent.absolute(),'replays')
  
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name=map_name,
                    players=[sc2_env.Agent(sc2_env.Race.terran, "Tergot"),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(
                            screen=96, minimap=64),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    realtime=False,
                    save_replay_episodes=0,
                    replay_dir=replaysDir,
                    visualize=False) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()
                score = 0
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        print("score:{}".format(float(score)))
                        episode_scores.append(float(score))
                        plot_scores(map_name)
                        break
                    timesteps = env.step(step_actions)
                    score += timesteps[0].reward
                    

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)