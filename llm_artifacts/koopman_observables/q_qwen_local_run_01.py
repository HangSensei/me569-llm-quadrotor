import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        # Control inputs
        x[6], x[7],
        # Trigonometric functions of theta
        math.sin(x[2]), math.cos(x[2]),
        # Control combinations
        x[6] + x[7], x[7] - x[6],
        # Products of velocity and trig functions
        x[3] * math.sin(x[2]), x[4] * math.cos(x[2]),
        x[3] * math.cos(x[2]), x[4] * math.sin(x[2]),
        # Higher-order trig terms
        math.sin(x[2]) ** 2, math.cos(x[2]) ** 2, math.sin(x[2]) * math.cos(x[2]),
        # Position-velocity products
        x[0] * x[3], x[1] * x[4],
        # Angular velocity products
        x[5] * math.sin(x[2]), x[5] * math.cos(x[2])
    ])

import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        # Trigonometric functions of theta
        math.sin(x[2]), math.cos(x[2]),
        # Products of velocity and trig functions
        x[3] * math.sin(x[2]), x[4] * math.cos(x[2]),
        x[3] * math.cos(x[2]), x[4] * math.sin(x[2]),
        # Higher-order trig terms
        math.sin(x[2]) ** 2, math.cos(x[2]) ** 2, math.sin(x[2]) * math.cos(x[2]),
        # Position-velocity products
        x[0] * x[3], x[1] * x[4],
        # Angular velocity products
        x[5] * math.sin(x[2]), x[5] * math.cos(x[2])
    ])

import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        # Trigonometric functions of theta
        np.sin(x[2]), np.cos(x[2]),
        # Products of velocity and trig functions
        x[3] * np.sin(x[2]), x[4] * np.cos(x[2]),
        x[3] * np.cos(x[2]), x[4] * np.sin(x[2]),
        # Higher-order trig terms
        np.sin(x[2]) ** 2, np.cos(x[2]) ** 2, np.sin(x[2]) * np.cos(x[2]),
        # Position-velocity products
        x[0] * x[3], x[1] * x[4],
        # Angular velocity products
        x[5] * np.sin(x[2]), x[5] * np.cos(x[2])
    ])

import numpy as np
import math

def observables(x):
    # The first 6 elements MUST be the state itself.
    return np.array([
        x[0], x[1], x[2], x[3], x[4], x[5],
        # Trigonometric functions of theta
        np.sin(x[2]), np.cos(x[2]),
        # Products of velocity and trig functions
        x[3] * np.sin(x[2]), x[4] * np.cos(x[2]),
        x[3] * np.cos(x[2]), x[4] * np.sin(x[2]),
        # Higher-order trig terms
        np.sin(x[2]) ** 2, np.cos(x[2]) ** 2, np.sin(x[2]) * np.cos(x[2]),
        # Position-velocity products
        x[0] * x[3], x[1] * x[4],
        # Angular velocity products
        x[5] * np.sin(x[2]), x[5] * np.cos(x[2])
    ])