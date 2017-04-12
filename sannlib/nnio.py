import numpy as np
import random
import json


def read_input(file_path, segment, shuffle_data=False):
    data = np.loadtxt(file_path)

    input_size = len(data[0])

    if shuffle_data:
        random.shuffle(data)

    inputs = np.asmatrix(data)[:, range(0, segment)]
    outputs = np.asmatrix(data)[:, range(segment, input_size)]

    return {"inputs": inputs, "outputs": outputs}


def merge_feature_and_class(feature_path, class_path):

    feature_data = np.loadtxt(feature_path)
    class_data = open(class_path).readlines()

    if len(feature_data) != len(class_data):
        print("feature and class data are illegal")
        exit(0)

    class_counter = 0
    class_list = dict()

    merged_data = []

    for i in range(0, len(feature_data)):

        if class_data[i] not in class_list:
            class_list[class_data[i]] = class_counter
            class_counter += 1

        merged_data.append({
            "input": np.asmatrix(feature_data[i]),
            "category": class_list[class_data[i]]
        })

    return merged_data


def read_training_and_test_data(filePath, segment, input_rows, shuffle_data=False):

    total_data = read_input(filePath, segment)

    if input_rows < 1:
        input_rows = input_rows * len(total_data["inputs"])

    training_data = dict()
    test_data = dict()

    all_inputs = total_data["inputs"]
    all_outputs = total_data["outputs"]

    training_data["inputs"] = all_inputs[0:input_rows, ::]
    test_data["inputs"] = all_inputs[input_rows:, ::]

    training_data["outputs"] = all_outputs[0:input_rows, ::]
    test_data["outputs"] = all_outputs[input_rows:, ::]

    return training_data, test_data


def save_result_to_json(path, nn_dict, note=None):

    if note:
        nn_dict["note"] = note

    with open(path, 'w') as fp:
        json.dump(nn_dict, fp)
