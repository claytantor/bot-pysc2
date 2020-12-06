import sys
import os
import logging
import random

from dotenv import load_dotenv, find_dotenv
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

        self.first(obs)
        actions_list = []
        actions_list.append(self.move_marine(obs))
        # actions_list.append(self.build_units(obs))
        # actions_list.append(self.attack(obs))

        #filter out empties
        active_actions = list(filter(lambda x: x != None, actions_list))

        if len(active_actions)==0:
            return actions.FUNCTIONS.no_op()
        else:
            return active_actions[0]
    
    def setup(self, obs_spec, action_spec):
        return super(TerranMoveToBeaconAgent, self).setup(obs_spec, action_spec)
    
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
