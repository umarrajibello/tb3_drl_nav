from env.tb3_env import TB3Env
from env.tb3_env_hybrid import TB3HybridEnv

def make_env(hybrid=False):
    if hybrid:
        return TB3HybridEnv()
    else:
        return TB3Env()
