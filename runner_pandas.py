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


def main(unused_argv):
    print("starting sc2 bot app.")

    load_dotenv(find_dotenv())
    
    agent = QTableAgent()

    replaysDir = os.path.join(Path(__file__).parent.absolute(),'replays')
  
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="MoveToBeacon",
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
                    save_replay_episodes=1,
                    replay_dir=replaysDir,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)