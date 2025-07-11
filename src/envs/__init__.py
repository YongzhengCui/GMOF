from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from smacv2.env import StarCraft2Env, StarCraftCapabilityEnvWrapper

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
