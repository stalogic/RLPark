from .ac import ActorCritic
from .dqn import DQN

def list_discrete_algorithms():
    return ['ActorCritic', 'DQN']

DISCRETE_ALGO_LIST = list_discrete_algorithms()


def list_continuous_algorithms():
    return ['DDPG']

CONTINUOUS_ALGO_LIST = list_continuous_algorithms()