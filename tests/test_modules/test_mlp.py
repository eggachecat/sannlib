import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '....', 'sannlib'))
import sannlib
import numpy as np
import pylab as plt

af_types = [("purelin", 1), ("tanh", 1), 10]
data_file_path = 'assets/test_mlp_.txt'
training_data, test_data = sannlib.nnio.read_training_and_test_data(data_file_path, 2, 950)

inputs = training_data["inputs"]
outputs = training_data["outputs"]

data = sannlib.nnio.read_input(data_file_path, 2)
test_inputs = data["inputs"]
test_outputs = data["outputs"]

alpha = 0.1

# cycle of data set
EPOCH = 1000
nn_config = {
    "alpha": alpha,
    "layers": [2, 4, 2, 1],
    "af_types": af_types
}

mlp = sannlib.mlp.MultipleLayerPerceptron(nn_config)

mlp_canvas = sannlib.nnplot.NeuralNetworkCanvas()
mlp_canvas.draw_classification_data_point_2d(np.loadtxt(data_file_path), 2)
# mlp_canvas.set_axis_lim((-1, 1), (-1, 1))


def calculate_error(_mlp_nn, _test_inputs, _test_outputs):
    _total_error = 0
    for _i in range(0, len(_test_inputs)):
        _input_vector = np.transpose(np.mat(_test_inputs[_i]))
        output = _mlp_nn.forward(_input_vector)
        teacher = np.transpose(np.mat(_test_outputs[_i]))
        flag = output * teacher > 0
        if flag[0, 0]:
            _total_error += 1
    return _total_error


for _ in range(0, EPOCH):
    # print(mlp.layers[1].weight)
    # print("-----------------------")
    # print(mlp.layers[1].bias)
    # print("=======================")
    for i in range(0, len(inputs)):
        input_vector = np.transpose(np.mat(inputs[i]))
        output_vector = mlp.forward(input_vector)
        teacher_vector = np.transpose(np.mat(outputs[i]))
        error_vector = teacher_vector - output_vector
        mlp.backpropagation(error_vector)
        # print(_, i)
    # mlp.update_neuron_plot_2d(mlp_canvas)
    # print(_, i)

    total_error = calculate_error(mlp, test_inputs, test_outputs)
    error_rate = float(total_error / len(test_inputs))
    print("{id}: {er}%".format(id=_, er=100 * error_rate))
#
# for _ in range(0, EPOCH):
#     # print(mlp.layers[1].weight)
#     # print("-----------------------")
#     # print(mlp.layers[1].bias)
#     # print("=======================")
#
#     error_vector_arr = []
#     abs_error_vector = np.array([[0.0]])
#     for i in range(0, len(inputs)):
#         input_vector = np.transpose(np.mat(inputs[i]))
#         output_vector = mlp.forward(input_vector)
#         teacher_vector = np.transpose(np.mat(outputs[i]))
#         error_vector = (teacher_vector - output_vector)
#         error_vector_arr.append(error_vector)
#         abs_error_vector += np.dot(np.transpose(error_vector), error_vector)
#
#     for error_vector in error_vector_arr:
#         print(error_vector)
#         mlp.backpropagation(error_vector)
#     # print(_, i)
#     print(mlp.layers[0].weight)
#     mlp.update_neuron_plot_2d(mlp_canvas)
#     # print(_, i)
#
#     total_error = calculate_error(mlp, test_inputs, test_outputs)
#     error_rate = float(total_error / len(test_inputs))
#     print("{id}: {er}%, abs: {abser}".format(id=_, er=100 * error_rate, abser=abs_error_vector))
