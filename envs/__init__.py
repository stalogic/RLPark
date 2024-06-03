from .mountain_car import mountain_car_raw as mountain_car_v0
from .mountain_car import mountain_car_reward_redefined as mountain_car_v1
from .mountain_car import mountain_car_state_reward_redefined as mountain_car_v2
from .poker_game import poker_game_raw as poker_game_v0
from .poker_game import poker_game_raw_2d as poker_game_2d_v0
from .cart_pole import cart_pole_v0
from .cart_pole import cart_pole_v1
from .acrobot import acrobot_v1

def list_discrete_envs():
    return [
        "mountain_car_v0",
        "mountain_car_v1",
        "mountain_car_v2",
        "poker_game_v0",
        # "poker_game_2d_v0",
        "cart_pole_v0",
        "cart_pole_v1",
        "acrobot_v1",
    ]

DISCRETE_ENV_LIST = list_discrete_envs()

from .pendulum import pendulum_v1
from .mountain_car_continuous import mountain_car_continuous_raw as mountain_car_continuous_v0

def list_continuous_envs():
    return [
        "pendulum_v1",
        "mountain_car_continuous_v0",
    ]

CONTINUOUS_ENV_LIST = list_continuous_envs()
