import numpy as np


def func_boha(x):
    """Return values of Bohachevsky functions.
    Args:
        - x(arrray): array of function arguments
    Return:
        - res(array): values of the functions
    """

    x1 = x[0]
    x2 = x[1]

    func_1 = (
        x1 ** 2
        + 2 * x2 ** 2
        - (0.3) * np.cos(3 * np.pi * x1)
        - (0.4) * np.cos(4 * np.pi * x2)
        + (0.7)
    )
    func_2 = (
        x1 ** 2
        + 2 * x2 ** 2
        - (0.3) * np.cos(3 * np.pi * x1) * np.cos(4 * np.pi * x2)
        + (0.3)
    )
    func_3 = (
        x1 ** 2 + 2 * x2 ** 2 - (0.3) * np.cos(3 * np.pi * x1 + 4 * np.pi * x2) + (0.3)
    )
    res = np.array([func_1, func_2, func_3])
    return res


def jacob_boha(x):
    """Return Jacobian matrix of Bohachevsky functions."""
    x1 = x[0]
    x2 = x[1]

    der_f1_x1 = 2 * x1 + 0.9 * np.pi * np.sin(3 * np.pi * x1)
    der_f1_x2 = 4 * x2 + 1.6 * np.pi * np.sin(4 * np.pi * x2)
    der_f2_x1 = 2 * x1 + 0.9 * np.pi * np.sin(3 * np.pi * x1) * np.cos(4 * np.pi * x2)
    der_f2_x2 = 4 * x2 + 1.2 * np.pi * np.cos(3 * np.pi * x1) * np.sin(4 * np.pi * x2)
    der_f3_x1 = 2 * x1 + 0.9 * np.pi * np.sin(3 * np.pi * x1 + 4 * np.pi * x2)
    der_f3_x2 = 4 * x2 + 1.2 * np.pi * np.sin(3 * np.pi * x1 + 4 * np.pi * x2)

    return np.array(
        [[der_f1_x1, der_f1_x2], [der_f2_x1, der_f2_x2], [der_f3_x1, der_f3_x2]]
    )


def hessian_boha(x):
    """Return Hessian matrix of Bohachevsky functions."""
    x1 = x[0]
    x2 = x[1]

    der_f1_x1_x1 = 2 + 2.7 * np.pi ** 2 * np.cos(3 * np.pi * x1)
    der_f1_x1_x2 = 0
    der_f1_x2_x1 = 0
    der_f1_x2_x2 = 4 + 6.4 * np.pi ** 2 * np.cos(4 * np.pi * x2)
    der_f2_x1_x1 = 2 + 2.7 * np.pi ** 2 * np.cos(3 * np.pi * x1) * np.cos(
        4 * np.pi * x2
    )
    der_f2_x1_x2 = -3.6 * np.pi ** 2 * np.sin(3 * np.pi * x1) * np.sin(4 * np.pi * x2)
    der_f2_x2_x1 = -3.6 * np.pi ** 2 * np.sin(3 * np.pi * x1) * np.sin(4 * np.pi * x2)
    der_f2_x2_x2 = 4 + 4.8 * np.pi ** 2 * np.cos(3 * np.pi * x1) * np.cos(
        4 * np.pi * x2
    )
    der_f3_x1_x1 = 2 + 2.7 * np.pi ** 2 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)
    der_f3_x1_x2 = 3.6 * np.pi ** 2 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)
    der_f3_x2_x1 = 3.6 * np.pi ** 2 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)
    der_f3_x2_x2 = 4 + 4.8 * np.pi ** 2 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)

    return np.array(
        [
            [[der_f1_x1_x1, der_f1_x1_x2], [der_f1_x2_x1, der_f1_x2_x2]],
            [[der_f2_x1_x1, der_f2_x1_x2], [der_f2_x2_x1, der_f2_x2_x2]],
            [[der_f3_x1_x1, der_f3_x1_x2], [der_f3_x2_x1, der_f3_x2_x2]],
        ]
    )
