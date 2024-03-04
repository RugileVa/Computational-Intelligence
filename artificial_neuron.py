import math
import numpy as np

inputs = np.array([[1, -0.2, 0.5],
                   [1, 0.2, -0.7],
                   [1, 0.8, -0.8],
                   [1, 0.8, 1]])

target_class = np.array([0, 0, 1, 1])

weight_interval = (-2, 2)

def step_function(a):
    return 1 if a > 0 else 0

def sigmoid_function(a):
    return round(1 / (1 + math.exp(-a)))

def activate(inputs, weights, func):
    h = 0
    for x, w in zip(inputs, weights):
        h += x * w

    return func(h)

def find_weights_randomly(inputs, target_class, func):
    weights = np.random.uniform(low=weight_interval[0], high=weight_interval[1], size=inputs.shape[1])
    outputs = np.zeros_like(target_class)  
    for i in range(len(inputs)):
        outputs[i] = activate(inputs[i], weights, func)  

    if np.array_equal(outputs, target_class):
        return weights
    else:
        return find_weights_randomly(inputs, target_class, func)

for i in range(1, 6):
    print(f"Step function weight set {i}: {find_weights_randomly(inputs, target_class, step_function)}")

for i in range(1, 6):
    print(f"Sigmoid function weight set {i}: {find_weights_randomly(inputs, target_class, sigmoid_function)}")
