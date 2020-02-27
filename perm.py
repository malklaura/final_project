import numpy as np


def func_perm(param, x):
    """Return value of Perm function in two dimensional form.
    Args:
        - param(float): value of parameter
        - x(arrray): array of function arguments
    Return:
        - func(float): value of the function
    """
    x1 = x[0]
    x2 = x[1]
    func = ((param + 1) * (x1 - 1) + (param + 2) * (x2 / 2 - 1)) ** 2 + (
        (param + 1) * (x1 ** 2 - 1) + (param + 4) * (x2 ** 2 / 4 - 1)
    ) ** 2
    return func


def grad_perm(param, x):
    """Return gradient vector of Perm function."""
    x1 = x[0]
    x2 = x[1]
    der_x1 = (param + 1) * (
        (4 * param + 4) * x1 ** 3
        + ((param + 4) * x2 ** 2 - 6 * param - 18) * x1
        + (param + 2) * x2
        - 4 * param
        - 6
    )
    der_x2 = (param + 4) * x2 * (
        (param + 4) * (x2 ** 2 / 4 - 1) + (param + 1) * (x1 ** 2 - 1)
    ) + (param + 2) * ((param + 2) * (x2 / 2 - 1) + (param + 1) * (x1 - 1))
    return np.array([der_x1, der_x2])


def hessian_perm(param, x):
    """Return Hessian matrix of Perm function."""
    x1 = x[0]
    x2 = x[1]

    der_x1_x1 = (param + 1) * (
        (12 * param + 12) * x1 ** 2 + (param + 4) * x2 ** 2 - 6 * param - 18
    )
    der_x1_x2 = (param + 1) * (2 * (param + 4) * x1 * x2 + param + 2)
    der_x2_x1 = (param + 1) * ((2 * param + 8) * x2 * x1 + param + 2)
    der_x2_x2 = (
        (3 * (param + 4) ** 2 * x2 ** 2) / 4
        + (param + 1) * (param + 4) * x1 ** 2
        - (3 * param ** 2 + 22 * param + 36) / 2
    )

    return np.array([[der_x1_x1, der_x1_x2], [der_x2_x1, der_x2_x2]])
