from .ac import OffPolicyActorCritic, OffPolicyActorCriticContinuous, ActorCritic, ActorCriticContinuous
from .dqn import DQN
from .ppo import PPO, PPOContinuous
from .ddpg import DDPG

def list_discrete_algorithms():
    return ['OffPolicyActorCritic', 'ActorCritic', 'DQN', 'PPO']

DISCRETE_ALGO_LIST = list_discrete_algorithms()


def list_continuous_algorithms():
    return ['OffPolicyActorCriticContinuous', 'ActorCriticContinuous', 'PPOContinuous', 'DDPG']

CONTINUOUS_ALGO_LIST = list_continuous_algorithms()