from typing import Callable, List, Tuple, Any, Type
from envs.base import Spec
from models.base import MultiPolicy

Done = bool
Reward = int
Action = List[Any]
Observation = List[Any]

PolicyType = Type[MultiPolicy]
ModelBuilder = Callable[[Spec, Spec], None]
