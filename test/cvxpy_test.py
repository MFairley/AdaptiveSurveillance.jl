# exec(open("cvxpy_test.py").read())
import numpy as np
import cvxpy as cp
from functools import partial
from itertools import product
from multiprocessing import Pool
from scipy.special import expit, softmax
from scipy.stats import binom
import matplotlib.pyplot as plt
import time
import dask

n = 200
W = np.array([1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0, 0.0, 4.0, 3.0, 2.0, 2.0, 
2.0, 4.0, 2.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 8.0, 2.0, 5.0, 
3.0, 0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 
2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 6.0, 2.0, 2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 
2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 5.0, 3.0, 0.0, 2.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 3.0, 
3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 4.0, 1.0, 1.0, 3.0, 1.0, 2.0, 4.0, 2.0, 1.0, 1.0, 
4.0, 1.0, 0.0, 2.0, 1.0, 5.0, 2.0, 3.0, 3.0, 4.0, 0.0, 6.0, 4.0, 3.0, 4.0, 3.0, 1.0, 4.0, 5.0, 5.0, 
1.0, 2.0, 5.0, 4.0, 4.0, 4.0, 7.0, 3.0, 2.0, 2.0, 7.0, 5.0, 3.0, 2.0, 3.0, 4.0, 4.0, 3.0, 6.0, 1.0, 
6.0, 4.0, 1.0, 4.0, 8.0, 2.0, 2.0, 3.0, 3.0, 2.0, 4.0, 3.0, 8.0, 2.0, 7.0, 7.0, 5.0, 8.0, 9.0, 6.0, 
8.0, 8.0, 4.0, 5.0, 8.0, 2.0, 8.0, 4.0, 6.0, 6.0, 13.0, 9.0, 12.0, 8.0, 6.0, 8.0, 4.0, 5.0, 9.0, 4.0, 
9.0, 8.0, 4.0, 7.0, 10.0, 11.0, 12.0, 8.0, 11.0, 9.0, 12.0, 12.0, 13.0, 13.0, 10.0, 10.0, 11.0, 15.0, 
12.0, 11.0, 4.0, 7.0, 7.0, 9.0, 13.0, 11.0, 13.0, 16.0, 11.0, 9.0, 16.0, 12.0, 11.0, 15.0, 11.0, 9.0, 
16.0, 11.0, 13.0, 16.0, 12.0, 17.0, 11.0, 20.0, 16.0, 16.0, 19.0, 15.0, 12.0, 13.0, 12.0, 14.0, 15.0, 
17.0, 22.0, 20.0, 16.0, 20.0, 17.0, 18.0, 17.0, 19.0, 16.0, 19.0, 17.0, 23.0, 24.0, 22.0, 19.0, 19.0, 
20.0, 19.0, 23.0, 28.0, 20.0, 20.0, 24.0, 26.0, 25.0, 26.0, 21.0, 34.0, 29.0, 28.0, 23.0, 29.0, 28.0, 
27.0, 36.0, 34.0, 29.0, 22.0, 17.0, 29.0, 28.0, 23.0, 39.0, 20.0, 28.0, 31.0, 23.0, 37.0, 31.0, 39.0, 
49.0])
t = np.arange(0, len(W))
t_up = 15
tp = t_up + 10

def create_problem(m, beta_max = 1.0, z_max = 0.0): # m: number of data points
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
    problem, Wp, qp, beta, z = create_problem(m)
    f = partial(solve_subproblem, problem, Wp, qp, beta, z, W, t, tp)
    # f(0, 0) # presolve for caching - this causes an assertion error
    with Pool(processes=8) as pool:
        M = pool.starmap(f, product(range(n+1), range(tmax+1)))
    # dasklist = [dask.delayed(f)(y, g) for (y, g) in product(range(n+1), range(tmax+1))]
    # dasklist = [dask.delayed(f)(y, g) for (y, g) in product(range(1), range(1))]
    # print(dasklist)
    # M = dask.compute(*dasklist, scheduler = 'processes')
    # M = 1
    # M = []
    # for (i, (y, g)) in enumerate(product(range(n+1), range(tmax+1))):
        # M.append(f(y, g))
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

if __name__ == '__main__':
    start = time.time()
    pl = profile_likelihood(n, W[0:t_up], t[0:t_up], tp)
    end = time.time()
    print("Time taken was {}".format(end - start))

    fig, ax = plt.subplots()
    ax.bar(np.arange(0, n + 1), pl)
    ax.set_xlabel('Number of Positive Tests')
    ax.set_ylabel('Probability')
    ax.set_title("Predictive Distribution for Time {} at Time {} \n BetaU = 1.0, zU = logit(0.5)".format(tp, t_up))
    fig.savefig("../results/tmp/cvx_pred_dist_{}samples_{}steps.pdf".format(t_up, tp - t_up))
    plt.close(fig)