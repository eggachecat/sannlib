import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '....', 'sannlib'))
import sannlib
import numpy as np
import time

config = {
    "acceleration_of_gravity": 9.8,
    "mass_of_cart": 1,
    "mass_of_pole": 0.1,
    "update_time_interval": 0.02,
    "half_length_of_pole": 0.5,
    "force": 10
}
cart_pole_instance = sannlib.nnsimulation.CartPole(config, figure=False)

degree_threshold = np.pi / 180
FAILURE_STATES_PARTITION = {
    "x": {
        "-1": lambda x: x < -2.4 or x > 2.4
    },
    "theta": {
        "-1": lambda x: x < -12 * degree_threshold or x > 12 * degree_threshold
    }
}

STATES_PARTITION = {
    "x": {
        "1": lambda x: x < -0.8,
        "2": lambda x: x < 0.8,
        "3": lambda x: x > 0.8
    },
    "v_x": {
        "0": lambda x: x < -0.5,
        "1": lambda x: x < 0.5,
        "2": lambda x: x > 0.5
    },
    "theta": {
        "0": lambda x: x < -6 * degree_threshold,
        "1": lambda x: x < -1 * degree_threshold,
        "2": lambda x: x < 0,
        "3": lambda x: x < degree_threshold,
        "4": lambda x: x < 6 * degree_threshold,
        "5": lambda x: True
    },
    "v_theta": {
        "0": lambda x: x < -50 * degree_threshold,
        "1": lambda x: x < 50 * degree_threshold,
        "2": lambda x: True
    }
}

greek = {
    "alpha": 0.5,
    "beta": 0.5,
    "gamma": 0.95,
    "eta": 0.8,
    "delta": 0.01
}

action_set = [1, -1]
test = 0
TOTAL_TESTS = 1000
state_variables = ["x", "v_x", "theta", "v_theta"]
q_learning_neural_network = sannlib.reinforecmentnn.QLearning(state_variables, STATES_PARTITION,
                                                              FAILURE_STATES_PARTITION, action_set, greek)
success_trails = 0
max_success_trails = 0

sv_records = []


def draw_stat(records):
    x_range = [r["x"] for r in records]
    v_x_range = [r["v_x"] for r in records]
    theta_range = [r["theta"] for r in records]
    v_theta_range = [r["v_theta"] for r in records]

    xv_canvas = sannlib.nnplot.NeuralNetworkCanvas(shape=(3, 1), dpi=200)

    xv_canvas.draw_line_chart_2d(theta_range, v_theta_range, sub_canvas_id=1)
    xv_canvas.set_label("pole angle", "pole angle velocity", 1)

    xv_canvas.draw_line_chart_2d(x_range, v_x_range, sub_canvas_id=2)
    xv_canvas.set_label("cart position", "cart velocity", 2)

    xv_canvas.draw_line_chart_2d(x_range, v_theta_range, sub_canvas_id=3)
    xv_canvas.set_label("cart position", "pole angle", 3)
    # xv_canvas.froze()

    xv_canvas.save("d:/report/test_{ts}-{fn}.png".format(ts=time.time(), fn=len(records)))


while test < TOTAL_TESTS:

    action = q_learning_neural_network.get_action()
    direction = action

    cart_pole_instance.update(direction)
    state_vector = cart_pole_instance.get_state()

    q_learning_neural_network.set_state(state_vector)

    sv_records.append(state_vector)

    if q_learning_neural_network.is_failed():
        cart_pole_instance.reset()

        if len(sv_records) > 500:
            draw_stat(sv_records)

        sv_records = []

        if success_trails > max_success_trails:
            max_success_trails = success_trails
        print("{id}: {trails}".format(id=test, trails=success_trails))
        success_trails = 0
        test += 1
    else:
        success_trails += 1

        # cart_pole_instance.draw(pause_interval=0.001,
        #                         title="Trail #{nt} Best: {mx}".format(nt=success_trails, mx=max_success_trails))
