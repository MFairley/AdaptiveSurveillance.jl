using Convex
using Mosek
using MosekTools

x = Variable()
y = Variable()
problem = minimize(0.0 * x + y)
solve!(problem, () -> Mosek.Optimizer())