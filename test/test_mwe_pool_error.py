# exec(open("cvxpy_test.py").read())
import numpy as np
import cvxpy as cp
from functools import partial
from itertools import product
from multiprocessing import Pool

n = 200
W = np.array([1.0, 1.0, 1.0, 2.0, 4.0, 4.0])
t = np.arange(0, len(W))
tmax = len(W)
m = len(W) + 1
tp = tmax + 10

def create_problem(m, beta_max = 1.0, z_max = 0.0): # m: number of data points
    W = cp.Parameter(m, nonneg=True)
    q = cp.Parameter(m, nonneg=True)
    beta = cp.Variable(nonneg=True)
    z = cp.Variable()
    coeff = cp.Variable(m)
    con = [beta <= beta_max, z <= z_max, coeff == beta * q + z]
    obj = cp.sum(cp.multiply(W, coeff) - n * cp.logistic(coeff))
    problem = cp.Problem(cp.Maximize(obj), constraints=con)
    return problem, W, q, beta, z

def solve_subproblem(problem, Wp, qp, beta, z, W, t, tp, y, gamma):
    Wp.value = np.append(W, y)
    qp.value = np.maximum(0, np.append(t, tp) - gamma)
    problem.solve(solver=cp.MOSEK, enforce_dpp=True)
    return problem.value, beta.value, z.value[()]

def solve_problem(n, W, t, tp, m, tmax):
    problem, Wp, qp, beta, z = create_problem(m)
    f = partial(solve_subproblem, problem, Wp, qp, beta, z, W, t, tp)
    f(0, 0) # presolve for caching - this causes an assertion error
    # with Pool(processes=8) as pool:
        # M = pool.starmap(f, product(range(n+1), range(tmax+1)))
    # return M

solve_problem(n, W, t, tp, m, tmax)