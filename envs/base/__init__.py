import platform
from .spec import Space, Spec
from .abc import Env
from .shm_multiproc import ShmMultiProcEnv
from .msg_multiproc import MsgMultiProcEnv

MultiProcEnv = ShmMultiProcEnv
print('platform: {}'.format(platform.system()))
if platform.system() == 'Windows':
    MultiProcEnv = MsgMultiProcEnv