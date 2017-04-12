from . import nnmodel
from . import activationfunction

import math
import pylab as pl
import numpy as np


class MultipleLayerPerceptronLayer(nnmodel.LayerModel):
    def __init__(self, previous_layer_size, this_layer_size, layer_activation_function):
        nnmodel.LayerModel.__init__(self, previous_layer_size, this_layer_size, layer_activation_function)


class MultipleLayerPerceptron(nnmodel.NeuralNetworkModel):
    type = "backprogation"

    def __init__(self, nn_config):

        activation_functions = activationfunction.generate_activation_functions_from_array(nn_config["af_types"])
        size_of_layers = nn_config["layers"]

        nnmodel.NeuralNetworkModel.__init__(self, size_of_layers, activation_functions)

        self.alpha = nn_config["alpha"]
        self.layers = [MultipleLayerPerceptronLayer(1, size_of_layers[0], activation_functions[0])]
        for i in range(1, self.depth):
            self.layers.append(
                MultipleLayerPerceptronLayer(size_of_layers[i - 1], size_of_layers[i], activation_functions[i]))

    def forward(self, input_vector):
        """
        
        :param input_vector: 
        :return: 
        """

        # set first layer equal to input <-> neuronList[0]
        self.layers[0].outputs = input_vector

        for i in range(1, self.depth):
            current_layer = self.layers[i]
            previous_layer = self.layers[i - 1]
            current_layer.receive_signal(previous_layer.outputs)

        return self.layers[self.output_layer_index].outputs

    def display_forward(self, input_vector):
        """
        
        :param input_vector: 
        :return: 
        """

        # set first layer equal to input <-> neuronList[0]
        self.layers[0].outputs = input_vector

        for i in range(1, self.depth):
            current_layer = self.layers[i]
            previous_layer = self.layers[i - 1]
            current_layer.display_receive_signal(previous_layer.outputs)

            print("preLayer.outputs\n", previous_layer.outputs)
            print("current_layer.neurons\n", current_layer.neurons)

        return self.layers[self.output_layer_index].outputs

    def update_neuron_plot_2d(self, canvas, layer_index=1, sub_canvas_id=1):
        canvas.draw_neuron_lines(self.layers[layer_index], sub_canvas_id)
        canvas.show(0.00000000000001)

    def backpropagation(self, error_vector):
        """
        
        :param error_vector: 
        :return: 
        """

        output_layer = self.layers[self.output_layer_index]
        sensitivity = -2 * np.dot(output_layer.get_diagonal_derivative_matrix(), error_vector)

        for i in reversed(range(0, self.output_layer_index)):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            next_sensitivity = np.copy(sensitivity)

            sensitivity = np.dot(
                np.dot(
                    current_layer.get_diagonal_derivative_matrix(), np.transpose(next_layer.weight))
                , next_sensitivity)

            next_layer.learn_weight(-1 * self.alpha * np.dot(next_sensitivity, np.transpose(current_layer.outputs)))
            next_layer.learn_bias(-1 * self.alpha * next_sensitivity)
