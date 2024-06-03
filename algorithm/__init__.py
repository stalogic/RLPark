from .ac import ActorCritic
from .dqn import DQN

def list_algorithms():
    return ['ActorCritic', 'DQN']

ALGO_LIST = list_algorithms()