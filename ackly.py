import numpy as np


def ackley(params, x):
    """Return value of Ackley function in two dimensional form.
    Args:
        - params(arrray): array of parameters
        - x(arrray): array of function arguments
    Return:
        - res(float): value of the function
    """
    a = params[0]
    b = params[1]
    c = params[2]
    x1 = x[0]
    x2 = x[1]
    sum1 = x1 ** 2 + x2 ** 2
    sum2 = np.cos(c * x1) + np.cos(c * x2)

    term1 = -a * np.exp(-b * (np.sqrt((1 / 2.0) * sum1)))
    term2 = -np.exp((1 / 2) * sum2)
    res = term1 + term2 + a + np.exp(1)
    return res


def grad_ackley(params, x):
    """Return gradient vector of Ackley function."""
    a = params[0]
    b = params[1]
    c = params[2]
    x1 = x[0]
    x2 = x[1]

    sum1 = x1 ** 2 + x2 ** 2
    sum2 = np.cos(c * x1) + np.cos(c * x2)

    term1 = a * b * x1 * np.exp(-b * np.sqrt((1 / 2.0) * sum1)) / np.sqrt(2 * sum1)
    term2 = c * (1 / 2.0) * np.sin(c * x1) * np.exp((1 / 2.0) * sum2)
    term3 = a * b * x2 * np.exp(-b * np.sqrt((1 / 2.0) * sum1)) / np.sqrt(2 * sum1)
    term4 = c * (1 / 2.0) * np.sin(c * x2) * np.exp((1 / 2.0) * sum2)

    return np.array([term1 + term2, term3 + term4])


def hessian_ackley(params, x):
    """Return Hessian matrix of Ackley function."""
    a = params[0]
    b = params[1]
    c = params[2]
    x1 = x[0]
    x2 = x[1]

    sum1 = x1 ** 2 + x2 ** 2
    sum2 = np.cos(c * x1) + np.cos(c * x2)

    term1 = (
        -a
        * b
        * (
            np.sqrt(2) * b * (x1 ** 2) * (sum1 ** (1.5))
            - 2 * ((x1 * x2) ** 2)
            - 2 * (x2 ** 4)
        )
        * np.exp(-b * np.sqrt((sum1) / 2))
        / (2 ** (1.5))
        * (sum1 ** (2.5))
    )
    term2 = (
        -(c ** (2))
        * (np.exp((0.5) * sum2))
        * ((np.sin(c * x1)) ** 2 - 2 * np.cos(c * x1))
        / 4
    )
    term3 = (
        -a
        * b
        * x1
        * x2
        * ((2 ** (0.5)) * b * (sum1 ** (1.5)) + 2 * (x1 ** 2) + 2 * (x2 ** 2))
        * np.exp(-b * ((sum1) / 2) ** (0.5))
    ) / ((2 ** (1.5)) * (sum1 ** (2.5)))
    term4 = -(c ** (2)) * np.sin(c * x1) * np.exp((0.5) * sum2) * np.sin(c * x2) / 4
    term5 = (
        -a
        * b
        * x1
        * x2
        * ((2 ** (0.5)) * b * (sum1 ** (1.5)) + 2 * (x1 ** 2) + 2 * (x2 ** 2))
        * np.exp(-b * ((sum1) / 2) ** (0.5))
    ) / ((2 ** (1.5)) * (sum1 ** (2.5)))
    term6 = -(c ** (2)) * np.sin(c * x2) * np.exp((0.5) * sum2) * np.sin(c * x1) / 4
    term7 = (
        -a
        * b
        * (
            np.sqrt(2) * b * (x2 ** 2) * (sum1 ** (1.5))
            - 2 * ((x1 * x2) ** 2)
            - 2 * (x2 ** 4)
        )
        * np.exp(-b * np.sqrt((sum1) / 2))
        / (2 ** (1.5))
        * (sum1 ** (2.5))
    )
    term8 = (
        -(c ** (2))
        * (np.exp((0.5) * sum2))
        * ((np.sin(c * x2)) ** 2 - 2 * np.cos(c * x2))
        / 4
    )

    return np.array([[term1 + term2, term3 + term4], [term5 + term6, term7 + term8]])
