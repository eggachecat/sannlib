import numpy as np


class ActivationFunction:
    def __init__(self, activate, derivative_in_y, derivative):
        self.activate = activate

        # derivative in y or x expression
        self.derivative_in_y = derivative_in_y
        self.derivative = derivative


def sigmoid(a=1):
    return ActivationFunction(
        lambda x: 1 / (1 + np.exp(-1 * a * x)), True,
        lambda y: a * y * (1 - y))


def tanh(a=1):
    return ActivationFunction(
        lambda x: (np.exp(a * x) - np.exp(-a * x)) / (np.exp(a * x) + np.exp(-a * x)), True,
        lambda y: a * (1 - y * y))


def purelin(a=1):
    return ActivationFunction(lambda x: a * x, False, lambda x: a)


def generate_activation_functions_from_array(af_arr):
    """
    
    :param af_arr: 
    :return: 
    """
    afs = []
    activation_function_map = globals()
    fun = None

    for item in af_arr:
        if type(item) is tuple:
            fun_name = item[0]
            fun_parameter = item[1]
            fun = activation_function_map[fun_name](fun_parameter)
            afs.append(fun)
        else:
            if (not type(item) is int) or (item < 0) or (not fun):
                print("something wrong with the activation-function-config")
                exit()

            for x in range(0, item):
                afs.append(fun)

    return afs
