import numpy as np


def func_branin(params, x):
    """Return value of Branin function.
    Args:
        - params(arrray): array of parameters
        - x(arrray): array of function arguments
    Return:
        - func(float): value of the function
    """

    a = params[0]
    b = params[1]
    c = params[2]
    r = params[3]
    s = params[4]
    t = params[5]
    x1 = x[0]
    x2 = x[1]
    func = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return func


def grad_branin(params, x):
    """Return gradient vector of Branin function."""
    a = params[0]
    b = params[1]
    c = params[2]
    r = params[3]
    s = params[4]
    t = params[5]
    x1 = x[0]
    x2 = x[1]
    der_x1 = 2 * a * (c - 2 * b * x1) * (-b * x1 ** 2 + c * x1 + x2 - r) - s * (
        1 - t
    ) * np.sin(x1)
    der_x2 = 2 * a * (x2 - b * x1 ** 2 + c * x1 - r)
    return np.array([der_x1, der_x2])


def hessian_branin(params, x):
    """Return the Hessian matrix of Branin function."""
    a = params[0]
    b = params[1]
    c = params[2]
    r = params[3]
    s = params[4]
    t = params[5]
    x1 = x[0]
    x2 = x[1]

    der_x1_x1 = (
        -s * (1 - t) * np.cos(x1)
        + 2 * a * (c - 2 * b * x1) ** 2
        - 4 * a * b * (-b * x1 ** 2 + c * x1 + x2 - r)
    )
    der_x1_x2 = 2 * a * (c - 2 * b * x1)
    der_x2_x1 = 2 * a * (c - 2 * b * x1)
    der_x2_x2 = 2 * a
    return np.array([[der_x1_x1, der_x1_x2], [der_x2_x1, der_x2_x2]])
