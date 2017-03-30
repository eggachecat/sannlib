import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '....', 'sannlib'))
import sannlib
import numpy as np

TOTAL_TRAILS = 100

trail = 0

config = {
    "acceleration_of_gravity": 9.8,
    "mass_of_cart": 1,
    "mass_of_pole": 0.1,
    "update_time_interval": 0.02,
    "half_length_of_pole": 0.5,
    "force": 10
}

cart_pole_instance = sannlib.nnsimulation.CartPole(config, figure=True)
while trail < TOTAL_TRAILS:
    cart_pole_instance.draw()

    direction = (-1) ** np.random.randint(0, 2)
    print(direction)
    cart_pole_instance.update(direction)

    cart_pole_instance.draw()
