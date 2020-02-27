""" Tests for functions, gradiants, hessians and Jacobians."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from ackly import ackley
from ackly import grad_ackley
from ackly import hessian_ackley
from branin import func_branin
from branin import grad_branin
from branin import hessian_branin
from perm import func_perm
from perm import grad_perm
from perm import hessian_perm
from bohachevsky import func_boha
from bohachevsky import jacob_boha
from bohachevsky import hessian_boha
from levy import func_levy
from levy import grad_levy
from levy import hessian_levy

@pytest.fixture
def set_up_ackley():
    out = {}
    out['params'] = np.array([0.2, 20, 2*np.pi])
    out['x'] = np.array([1, 1])
    return out

def test_function_ackley(set_up_ackley):
    exp = 0.2
    res = ackley(**set_up_ackley)
    aaae(exp, res)

def test_function_ackley_grad(set_up_ackley):
    exp = [4.1223e-10, 4.1223e-10]
    res = grad_ackley(**set_up_ackley)
    aaae(exp, res)

def test_function_ackley_hess(set_up_ackley):
    exp = [[5.365673e+01, -4.32842261e-8],[ -4.32842261e-8,5.365673e+01]]
    res = hessian_ackley(**set_up_ackley)
    aaae(exp, res)

####branin

@pytest.fixture
def set_up_branin():
    out = {}
    out['params'] = np.array([1, 5.1/(4*(np.pi**2)), 5/np.pi, 6, 10, 1/(8*np.pi)])
    out['x'] = np.array([1, 1])
    return out

def test_function_branin(set_up_branin):
    exp = 27.7029055485
    res = func_branin(**set_up_branin)
    aaae(exp, res)

def test_function_branin_grad(set_up_branin):
    exp = [-17.512511, -7.07527]
    res = grad_branin(**set_up_branin)
    aaae(exp, res)

def test_function_branin_hess(set_up_branin):
    exp = [[0.194727, 2.666361], [2.666361, 2]]
    res = hessian_branin(**set_up_branin)
    aaae(exp, res)

####perm
@pytest.fixture
def set_up_perm():
    out = {}
    out['param'] = 0.5
    out['x'] = np.array([1, 1])
    return out

def test_function_perm(set_up_perm):
    exp = 12.953125
    res = func_perm(**set_up_perm)
    aaae(exp, res)

def test_function_perm_grad(set_up_perm):
    exp = np.array([-24, -18.3125])
    res = grad_perm(**set_up_perm)
    aaae(exp, res)

def test_function_perm_hess(set_up_perm):
    exp =[[2.25, 17.25], [17.25, -1.9375]]
    res = hessian_perm(**set_up_perm)
    aaae(exp, res)

####bohachevsky
@pytest.fixture
def set_up_boha():
    out = {}
    out['x'] = np.array([0.5, 1.7])
    return out

def test_function_boha(set_up_boha):
    exp = ([7.0536068, 6.33, 6.15366442])
    res = func_boha(**set_up_boha)
    aaae(exp, res)

def test_function_boha_jacob(set_up_boha):
    exp = ([[-1.82743339, 9.75453093],
            [ 3.28744166, 6.8],
            [ 3.28744166, 9.84992222]])
    res = jacob_boha(**set_up_boha)
    aaae(exp, res)

def test_function_boha_hess(set_up_boha):
    exp = ([[[2, 0], [0, -47.1019372]],
            [[2, 20.88434849], [20.88434849, 4]],
            [[17.66326136, 20.88434849], [20.88434849, 31.84579798]]])
    res = hessian_boha(**set_up_boha)
    aaae(exp, res)

####levy
@pytest.fixture
def set_up_levy():
    out = {}
    out['x'] = np.array([1, 3])
    return out

def test_function_levy(set_up_levy):
    exp = 4
    res = func_levy(**set_up_levy)
    aaae(exp, res)

def test_function_levy_grad(set_up_levy):
    exp = [0, 4]
    res = grad_levy(**set_up_levy)
    aaae(exp, res)

def test_function_levy_hess(set_up_levy):
    exp =[[179.65287922, 0],[0, 317.82734083]]
    res = hessian_levy(**set_up_levy)
    aaae(exp, res)
