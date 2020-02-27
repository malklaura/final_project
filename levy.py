import numpy as np


def func_levy(x):
    """Return value of Levy function N13.
    Args:
        - x(arrray): array of function arguments
    Return:
        - func(float): value of the function
    """
    x1 = x[0]
    x2 = x[1]
    func = (
        (np.sin(3 * np.pi * x1)) ** 2
        + (x2 - 1) ** 2 * (1 + np.sin(2 * np.pi * x2) ** 2)
        + (x1 - 1) ** 2 * (1 + np.sin(3 * np.pi * x2) ** 2)
    )
    return func


def grad_levy(x):
    """Return gradient vector of Levy function N13."""
    x1 = x[0]
    x2 = x[1]
    der_x1 = 6 * np.pi * np.cos(3 * np.pi * x1) * np.sin(3 * np.pi * x1) + 2 * (
        (np.sin(3 * np.pi * x2)) ** 2 + 1
    ) * (x1 - 1)
    der_x2 = (
        6 * np.pi * (x1 - 1) ** 2 * np.cos(3 * np.pi * x2) * np.sin(3 * np.pi * x2)
        + 2 * (x2 - 1) * ((np.sin(2 * np.pi * x2)) ** 2 + 1)
        + 4 * np.pi * (x2 - 1) ** 2 * np.cos(2 * np.pi * x2) * np.sin(2 * np.pi * x2)
    )
    return np.array([der_x1, der_x2])


def hessian_levy(x):
    """Return the Hessian matrix of Levy function N13."""
    x1 = x[0]
    x2 = x[1]

    der_x1_x1 = (
        -18 * np.pi ** 2 * (np.sin(3 * np.pi * x1)) ** 2
        + 18 * np.pi ** 2 * (np.cos(3 * np.pi * x1)) ** 2
        + 2 * ((np.sin(3 * np.pi * x2)) ** 2 + 1)
    )
    der_x1_x2 = 12 * np.pi * (x1 - 1) * np.cos(3 * np.pi * x2) * np.sin(3 * np.pi * x2)
    der_x2_x1 = 12 * np.pi * (x1 - 1) * np.cos(3 * np.pi * x2) * np.sin(3 * np.pi * x2)
    der_x2_x2 = (
        18 * np.pi ** 2 * (x1 - 1) ** 2 * np.cos(6 * np.pi * x2)
        + 8 * np.pi * (x2 - 1) * np.sin(4 * np.pi * x2)
        + (8 * np.pi ** 2 * (x2 - 1) ** 2 - 1) * np.cos(4 * np.pi * x2)
        + 3
    )

    return np.array([[der_x1_x1, der_x1_x2], [der_x2_x1, der_x2_x2]])
