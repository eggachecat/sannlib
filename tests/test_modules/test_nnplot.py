import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '....', 'sannlib'))
import sannlib


import numpy as np

data = np.loadtxt("assets/test_nnplot.txt")
canvas = sannlib.nnplot.NeuralNetworkCanvas("data image")
canvas.draw_data_point_2d(data, 2)

line_blue = canvas.draw_line_2d(0.5, 0.3, "b")
line_green = canvas.draw_line_2d(0.3, 0.2, "g")

canvas.show()
canvas.remove_line_2d(line_blue)
canvas.show()
