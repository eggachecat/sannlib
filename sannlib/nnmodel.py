import numpy as np


class MatrixOperationModel:
    def __init__(self):
        pass

    @staticmethod
    def apply(matrix, fn):
        for x in np.nditer(matrix, op_flags=['readwrite']):
            x[...] = fn(x)

    @staticmethod
    def pipe(matrix, fn):
        new_matrix = np.copy(matrix)
        for x in np.nditer(new_matrix, op_flags=['readwrite']):
            x[...] = fn(x)

        return new_matrix


class LayerModel:
    def __init__(self, previous_layer_size, this_layer_size, layer_activation_function):
        """
        
        :param previous_layer_size: 
        :param this_layer_size: 
        :param layer_activation_function: 
        """
        self.size = this_layer_size
        self.neurons = np.zeros((this_layer_size, 1), dtype=float)
        self.outputs = np.zeros((this_layer_size, 1), dtype=float)

        magic_number = np.sqrt(float(6) / float(previous_layer_size + this_layer_size))
        self.weight = (np.random.random(
            (this_layer_size, previous_layer_size)) - 0.5) * magic_number

        self.bias = np.zeros((this_layer_size, 1), dtype=float)

        self.layer_activation_function = layer_activation_function

    def receive_signal(self, signals):
        self.neurons = np.dot(self.weight, signals) + self.bias

        self.outputs = MatrixOperationModel.pipe(self.neurons, self.layer_activation_function.activate)

    def display_receive_signal(self, signals):
        self.neurons = np.dot(self.weight, signals) + self.bias
        print("self.weight\n", self.weight)
        print("self.neurons\n", self.neurons)
        self.outputs = MatrixOperationModel.pipe(self.neurons, self.layer_activation_function.activate)

    def get_diagonal_derivative_matrix(self):

        diagonal_derivative_matrix = None

        if self.layer_activation_function.derivative_in_y:
            diagonal_derivative_matrix = np.copy(self.outputs)
            MatrixOperationModel.apply(diagonal_derivative_matrix, self.layer_activation_function.derivative)
        else:
            diagonal_derivative_matrix = np.copy(self.neurons)
            MatrixOperationModel.apply(diagonal_derivative_matrix, self.layer_activation_function.derivative)

        if diagonal_derivative_matrix.shape == (1, 1):
            return diagonal_derivative_matrix

        return np.diag(np.squeeze(np.asarray(diagonal_derivative_matrix)))

    def learn_weight(self, adjust):
        self.weight = self.weight + adjust

    def learn_bias(self, adjust):
        self.bias = self.bias + adjust


class NeuralNetworkModel:
    """

    """

    def __init__(self, layers_config, activation_functions):
        """ sizeOfLayers 
            #0 -> input size

            #last -> output size
        """

        # L
        self.depth = len(layers_config)
        self.output_layer_index = self.depth - 1

        self.layers_config = layers_config
        self.activation_functions = activation_functions
