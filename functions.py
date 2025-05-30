import numpy as np

# Function 1: Exponentially decreases then increases (symmetric tanh)
def f1(x):
    return -np.tanh(2 * x)

# Function 2: Exponential rise and fall, then linear
def f2(x):
    return np.piecewise(x, [x <= 1, x > 1], [lambda x: np.tanh(2 * x), lambda x: 0.07 * x + 0.894])

# Function 3: Linearly decreases and increases, smoothed with tanh near joins
def f3(x):
    return np.piecewise(x, [x <= 0, x > 0], [lambda x: -0.5 * x - 0.5 * np.tanh(3 * (x + 0.5)), lambda x: 0.5 * x - 0.5 * np.tanh(3 * (x - 0.5))])