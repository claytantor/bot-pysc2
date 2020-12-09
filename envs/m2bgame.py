import gym
import numpy as np
import logging
from numpy.core.einsumfunc import _parse_possible_contraction
import pygame
from gym import spaces


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class SC2GymEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self,         
        map_name='MoveToBeacon',
        realtime=False,
        visualize=True,
        replay_dir='replays'):

    super(SC2GymEnv, self).__init__()

    # initialize the pygame module
    pygame.init()
    # load and set the logo
    logo = pygame.image.load("../assets/logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("move to beacon")
     
    # create a surface on screen that has the size of 240 x 180
    screen = pygame.display.set_mode((128,128))

    BLACK=(0,0,0)
    screen.fill(BLACK)

    # load image (it is in same directory)
    image = pygame.image.load("../assets/person.png")   
    # blit image to screen
    screen.blit(image, (0,0))

        # load image (it is in same directory)
    image2 = pygame.image.load("../assets/goal.png")   
    # blit image to screen
    screen.blit(image2, (96,96))


    # update the screen to make the changes visible (fullscreen update)
    pygame.display.flip()
  
    self._episode = 0
    self._num_step = 0
    self._episode_reward = 0
    self._total_reward = 0 

  def step(self, action):
      pass
    

  def reset(self):
      pass


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





