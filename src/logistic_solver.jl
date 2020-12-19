using Optim, NLSolversBase, Random
using StatsBase
import Convex, Mosek, MosekTools

### Optim

function log_likelihood(x, W, tΓ, n)
    # x: [beta, z]
    coeff = x[1] .* tΓ .+ x[2]
    return -sum(W .* coeff .- n .* logistic.(coeff))
end

function log_likelihood_grad!(g, x, W, tΓ, n)
    coeff = x[1] .* tΓ .+ x[2]
    sigd1 = logistic.(coeff) .* (1 .- logistic.(coeff)) 
    g[1] = -sum(W .* tΓ .- n .* sigd1 .* tΓ)
    g[2] = -sum(W .- n .* sigd1)
end

function log_likelihood_hess!(h, x, W, tΓ, n)
    coeff = x[1] .* tΓ .+ x[2]
    sigd2 = logistic.(coeff) .* (1 .- logistic.(coeff)) .* (1 .- 2 .* logistic.(coeff))
    h[1, 1] = -sum(n .* (sigd2 .* tΓ.^2 + sigd2))
    h[1, 2] = -sum(-n .* tΓ .* sigd2)
    h[2, 1] = -sum(-n .* tΓ .* sigd2)
    h[2, 2] = -sum(-n .* sigd2)
end

function solve_logistic_optim(W, t, Γ, n, x0 = [0.01, logit(0.01)], ux = [1.0, logit(0.5)])
    tΓ = max.(0, t .- Γ)
    
    fun = (x) -> log_likelihood(x, W, tΓ, n)
    fun_grad! = (g, x) -> log_likelihood_grad!(g, x, W, tΓ, n)
    fun_hess! = (h, x) -> log_likelihood_hess!(h, x, W, tΓ, n)
    
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    # df = TwiceDifferentiable(fun, x0, autodiff=:forward)
    dfc = TwiceDifferentiableConstraints([0.0, -Inf], ux)
    println("Starting optim...")
    res = optimize(df, dfc, x0, IPNewton())
    # res = optimize(df, x0)

    return -Optim.minimum(res), Optim.minimizer(res)[1], logistic(Optim.minimizer(res)[2])
end

### Convex.jl
function solve_logistic_convex(W, t, Γ, n, ux = [1.0, logit(0.5)])
    tΓ = max.(0, t .- Γ)
    β = Convex.Variable(Convex.Positive())
    z = Convex.Variable()
    coeff = β * tΓ + z
    obj = Convex.dot(W, coeff) - n * Convex.logisticloss(coeff)
    problem = Convex.maximize(obj, β <= ux[1], z <= ux[2])
    Convex.solve!(problem, () -> Mosek.Optimizer(QUIET=true), verbose=false)
    return problem.optval, Convex.evaluate(β), logistic(Convex.evaluate(z))
end