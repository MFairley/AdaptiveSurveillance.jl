import numpy as np
import cvxpy as cp
from functools import partial
from itertools import product
from multiprocessing import Pool
from scipy.special import softmax, expit, logit
from scipy.stats import binom
import time

N_PATHS = 1000
T_MAX = 100
N_LOCATIONS = 2
N_TESTS = 200
BETA = 0.015008
PO = 0.01
# random seed

# functions
# dimensions to learn: gamma, predicted time out, missing data, current time 
# n, W, t, tp
# generate data
# append to list
# run optim
# save to file
def replication(t_max, n_locations, n_tests, beta, p0, gamma):
    # profile_likelihood(n, W, t, tp):
    pass

def sample_count(t, n, beta, p0, gamma):
    p = expit(beta * np.maximum(0, t - gamma) + logit(p0))
    return binom.rvs(n, p)

def create_problem(n, m, beta_max = 1.0, z_max = 0.0): # m: number of data points
    W = cp.Parameter(m, nonneg=True) # test results
    q = cp.Parameter(m, nonneg=True) # cp.pos(t - gamma), need to precompute outside problem
    beta = cp.Variable(nonneg=True) # transition rate
    z = cp.Variable() # logit of initial prevalance
    coeff = cp.Variable(m) # this coeff variable is needed to make DPP
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
    problem, Wp, qp, beta, z = create_problem(n, m)
    f = partial(solve_subproblem, problem, Wp, qp, beta, z, W, t, tp)
    # f(0, 0) # presolve for caching - this causes an assertion error
    with Pool(processes=8) as pool:
        M = pool.starmap(f, product(range(n+1), range(tmax+1)))
    return M

def maximize_gamma(M, n, tmax):
    ll = np.ones(n + 1) * - np.Inf
    beta = np.zeros(n + 1)
    z = np.zeros(n + 1)
    gamma = np.zeros(n + 1)
    for (i, (y, g)) in enumerate(product(range(n+1), range(tmax+1))):
        if ll[y] <= M[i][0]:
            ll[y] = M[i][0]
            beta[y] = M[i][1]
            z[y] = M[i][2]
            gamma[y] = g
    return beta, z, gamma

def loglikelihood(n, W, t, beta, z, gamma):
    p = expit(beta * np.maximum(0, t - gamma) + z)
    return sum(binom.logpmf(W[i], n, p[i]) for i in range(len(t)))

def profile_likelihood(n, W, t, tp):
    m = len(W) + 1
    tmax = np.max(t)
    M = solve_problem(n, W, t, tp, m, tmax)
    beta, z, gamma = maximize_gamma(M, n, tmax)

    logl = np.zeros(n + 1)
    for i in range(n + 1):
        logl[i] = loglikelihood(n, np.append(W, i), np.append(t, tp), beta[i], z[i], gamma[i])
    
    return softmax(logl)

if __name__ == "__main__":
    print("Hello World")