from .ac import OffPolicyActorCritic, OffPolicyContinuousActorCritic, ActorCritic, ContinuousActorCritic
from .dqn import DQN

def list_discrete_algorithms():
    return ['OffPolicyActorCritic', 'ActorCritic', 'DQN']

DISCRETE_ALGO_LIST = list_discrete_algorithms()


def list_continuous_algorithms():
    return ['OffPolicyContinuousActorCritic', 'ContinuousActorCritic']

CONTINUOUS_ALGO_LIST = list_continuous_algorithms()