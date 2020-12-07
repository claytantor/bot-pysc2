import gym
import numpy as np
import logging
# import pygame

from gym import spaces
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.env.environment import StepType


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NO_OP = actions.FUNCTIONS.no_op.id


class SC2GymEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self,         
        map_name='MoveToBeacon',
        realtime=False,
        visualize=True,
        replay_dir='replays'):

    super(SC2GymEnv, self).__init__()

    # pygame.init()

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:

    # self.available_actions = [ actions.FUNCTIONS.no_op ]


    # # Example for using image as input:
    # self.observation_space = spaces.Box(low=0, high=255, shape=
    #                 (64, 64, 1), dtype=np.uint8)

    # metadata = {'render.modes': [None, 'human']}
    # default_settings = {'agent_interface_format': sc2_env.parse_agent_interface_format(
    #     feature_screen=84,
    #     feature_minimap=64,
    # )}

    self._env = sc2_env.SC2Env(
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
      realtime=realtime,
      save_replay_episodes=1,
      replay_dir=replay_dir,
      visualize=visualize)
      
    self._episode = 0
    self._num_step = 0
    self._episode_reward = 0
    self._total_reward = 0 

  def step(self, action):

    try:
        obs = self._env.step(action)[0]
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    except KeyboardInterrupt:
        logger.info("Interrupted. Quitting...")
        return None, 0, True, {}
    

  def reset(self):
      if self._episode > 0:
          logger.info("Episode %d ended with reward %d after %d steps.",
                      self._episode, self._episode_reward, self._num_step)
          logger.info("Got %d total reward so far, with an average reward of %g per episode",
                      self._total_reward, float(self._total_reward) / self._episode)
      self._episode += 1
      self._num_step = 0
      self._episode_reward = 0
      logger.info("Episode %d starting...", self._episode)
      obs = self._env.reset()[0]
      self.available_actions = obs.observation['available_actions']
      return obs

  def save_replay(self, replay_dir):
      self._env.save_replay(replay_dir)

  def close(self):
      if self._episode > 0:
          logger.info("Episode %d ended with reward %d after %d steps.",
                      self._episode, self._episode_reward, self._num_step)
          logger.info("Got %d total reward, with an average reward of %g per episode",
                      self._total_reward, float(self._total_reward) / self._episode)
      if self._env is not None:
          self._env.close()
      super().close()


  @property
  def action_spec(self):
      return self._env.action_spec()

  @property
  def observation_spec(self):
      return self._env.observation_spec()

  @property
  def episode(self):
      return self._episode

  @property
  def num_step(self):
      return self._num_step

  @property
  def episode_reward(self):
      return self._episode_reward

  @property
  def total_reward(self):
      return self._total_reward

  def render(self, mode='rgb_array', close=False):
    pass
    # Render the environment to the screen
    # env.render(mode='rgb_array').transpose((2, 0, 1))
    # Set up the drawing window
    # screen = pygame.display.set_mode([64, 64])
    # if mode=='rgb_array':
    #     imgdata = pygame.surfarray.array3d(screen)
    #     imgdata.swapaxes(0,1)
    #     return imgdata




