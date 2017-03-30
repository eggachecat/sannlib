import numpy as np
from . import nnplot


class CartPole:
    CART_LEVEL = 0.0
    CART_WIDTH = 1
    CART_HEIGHT = 0.3
    POLE_HEIGHT = CART_LEVEL + CART_HEIGHT
    """docstring for CartPole"""

    def __init__(self, config, figure=False, beta=0.01):
        self.g = config["acceleration_of_gravity"]
        self.m_c = config["mass_of_cart"]
        self.m_p = config["mass_of_pole"]
        self.m = self.m_c + self.m_p

        self.t = config["update_time_interval"]

        # half length of pole
        self.hl = config["half_length_of_pole"]
        self.f = config["force"]
        self.beta = beta

        # define state parameters
        self.v_theta = None
        self.theta = None
        self.x = None
        self.v_x = None

        # initialize state parameters
        self.ini_state()

        # initialize canvas elements
        self.pole = None
        self.cart = None

        self.cart_canvas = nnplot.NeuralNetworkCanvas("cart-pole simulation")

        if figure:
            self.ini_figure()

    def reset(self):
        self.ini_state()

    def ini_state(self):
        """
        
        :return: 
        """
        self.v_theta = self.beta * np.random.random_sample()
        self.theta = self.beta * np.random.random_sample()
        self.x = self.beta * np.random.random_sample()
        self.v_x = self.beta * np.random.random_sample()

    def ini_figure(self):
        """
        
        :return: 
        """
        self.x = 0.0
        self.theta = 0.0
        self.cart_canvas.clean_canvas()

        self.cart_canvas.set_axis_lim([-3, 3], [0, 3])
        self.draw()

    def get_state(self):
        return {
            "x": self.x,
            "v_x": self.v_x,
            "theta": self.theta,
            "v_theta": self.v_theta
        }

    def update(self, direction=1):
        """
        
        :param direction : (int) value from {-1, 1} -1 -> left and 1 -> right 
        :return: 
        """
        acceleration_of_theta = ((self.m * self.g * np.sin(self.theta) -
                                  np.cos(self.theta) * (
                                      direction * self.f + self.m_p * self.hl * np.power(self.v_theta, 2) * np.sin(
                                          self.theta))) /
                                 ((4 / 3) * self.m * self.hl - self.m_p * self.hl * np.power(np.cos(self.theta), 2)))

        acceleration_of_x = (direction * self.f + self.m_p * self.hl * (
            np.power(self.v_theta, 2) * np.sin(self.theta) - acceleration_of_theta * np.cos(self.theta))) / self.m;

        self.v_theta += acceleration_of_theta * self.t
        self.v_x += acceleration_of_x * self.t

        self.theta += self.v_theta * self.t
        self.x += self.v_x * self.t

        # self.set_cart(self.x)
        # self.set_pole(self.x, self.theta)

    def draw(self, title=None, pause_interval=0.01):

        print("call me")
        self.cart_canvas.clean_canvas()

        self.cart_canvas.add_line_2d([self.x, self.x + np.cos(np.pi / 2 - self.theta)],
                                     [self.POLE_HEIGHT, self.POLE_HEIGHT + np.sin(np.pi / 2 - self.theta)],
                                     color="red", lw=1.5)
        self.cart_canvas.add_patch_rectangle((self.x - self.CART_WIDTH / 2, self.CART_LEVEL), self.CART_WIDTH,
                                             self.CART_HEIGHT)
        if title:
            self.cart_canvas.set_title(title)

        self.cart_canvas.show(pause_interval)
