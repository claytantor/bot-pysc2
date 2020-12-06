from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def get_action(self, obs): ...

from agents.base.memory import MemoryAgent
from agents.base.running import SyncRunningAgent
from agents.base.actor_critic import ActorCriticAgent, DEFAULTS
