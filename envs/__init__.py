
from .classic_control import mountain_car_v0, mountain_car_v1, mountain_car_v2, mountain_car_v3, cart_pole_v0, cart_pole_v1, acrobot_v1


DISCRETE_ENV_LIST = [
        "mountain_car_v0",
        "mountain_car_v1",
        "mountain_car_v2",
        "mountain_car_v3",
        "cart_pole_v0",
        "cart_pole_v1",
        "acrobot_v1",
    ]


from .atari_game import pong_v0, pong_v1, breakout_v0

DISCRETE_2D_ENV_LIST = [
    # atari games
    "pong_v0",
    "pong_v1",
    "breakout_v0",
]


from .classic_control import pendulum_v1, pendulum_v2, mountain_car_continuous_v0, mountain_car_continuous_v1, mountain_car_continuous_v2, mountain_car_continuous_v3


CONTINUOUS_ENV_LIST = [
        # "pendulum_v1",
        "pendulum_v2",
        # "mountain_car_continuous_v0",
        "mountain_car_continuous_v1",
        "mountain_car_continuous_v2",
        "mountain_car_continuous_v3"
    ]
