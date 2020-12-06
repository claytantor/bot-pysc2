import sys
import os
from dotenv import load_dotenv, find_dotenv


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app


import random


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()

        self.attack_coordinates = None

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
        super(TerranAgent, self).step(obs)

        self.first(obs)
        actions_list = []
        actions_list.append(self.build_buildings(obs))
        actions_list.append(self.build_units(obs))
        actions_list.append(self.attack(obs))

        #filter out empties
        active_actions = list(filter(lambda x: x != None, actions_list))

        if len(active_actions)==0:
            return actions.FUNCTIONS.no_op()
        else:
            return active_actions[0]
    
    def first(self, obs):
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)
    
    def build_buildings(self, obs):

        supply_depot = self.get_units_by_type(obs, units.Terran.SupplyDepot)
        
        if len(supply_depot) == 0:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y)) 
            

            scvs = self.get_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                scv = random.choice(scvs)

                return actions.FUNCTIONS.select_point("select_all_type", (scv.x,
                                                                          scv.y))

        else:
            barracks = self.get_units_by_type(obs, units.Terran.Barracks)   
            if len(barracks) == 0:
                if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_Barracks_screen("now", (x, y)) 

        return None

    def build_units(self, obs):
        barracks = self.get_units_by_type(obs, units.Terran.Barracks)
        marines = self.get_units_by_type(obs, units.Terran.Marine)

        if len(barracks) > 0 and len(marines) < 11:
            if self.unit_type_is_selected(obs, units.Terran.Barracks):
                if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                    return actions.FUNCTIONS.Train_Marine_quick("now")
            else:
                barrack = random.choice(barracks)
                return actions.FUNCTIONS.select_point("select_all_type", (barrack.x,
                                                                          barrack.y))
      
        return None


    def attack(self, obs):

        marines = self.get_units_by_type(obs, units.Terran.Marine)
        if len(marines) >= 10:
            if self.unit_type_is_selected(obs, units.Terran.Marine):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now",
                                                            self.attack_coordinates)

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")   

        return None


