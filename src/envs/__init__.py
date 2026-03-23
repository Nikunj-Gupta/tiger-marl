from functools import partial
from .multiagentenv import MultiAgentEnv
from .gather import GatherEnv
from .gymma import GymmaWrapper


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)

REGISTRY = {}
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)
REGISTRY["gymma"] = gymma_fn


