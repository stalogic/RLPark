
from .classic_control import mountain_car_v0, mountain_car_v1, mountain_car_v2, mountain_car_v3, cart_pole_v0, cart_pole_v1, acrobot_v1

def list_discrete_envs():
    return [
        # "mountain_car_v0",
        "mountain_car_v1",
        "mountain_car_v2",
        "mountain_car_v3",
        # "cart_pole_v0",
        "cart_pole_v1",
        "acrobot_v1",
    ]

DISCRETE_ENV_LIST = list_discrete_envs()


from .classic_control import pendulum_v1, pendulum_v2, mountain_car_continuous_v0, mountain_car_continuous_v1, mountain_car_continuous_v2, mountain_car_continuous_v3

def list_continuous_envs():
    return [
        # "pendulum_v1",
        "pendulum_v2",
        # "mountain_car_continuous_v0",
        "mountain_car_continuous_v1",
        "mountain_car_continuous_v2",
        "mountain_car_continuous_v3"
    ]

CONTINUOUS_ENV_LIST = list_continuous_envs()
