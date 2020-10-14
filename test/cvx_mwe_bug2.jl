using Convex
using Mosek
using MosekTools
using SCS
W = [1, 0]
x = Variable()
y = Variable()
coeff = x * [1, 0] + y
obj = dot(W, coeff) - logisticloss(coeff)
problem = maximize(obj)

solve!(problem, () -> Mosek.Optimizer(QUIET=true), verbose=false, warmstart = true)
solve!(problem, () -> Mosek.Optimizer(QUIET=true), verbose=false, warmstart = true) # fails

# solve!(problem, () -> SCS.Optimizer(), verbose=false, warmstart = true) 
# solve!(problem, () -> SCS.Optimizer(), verbose=false, warmstart = true) # does not fail
