using Random
using Optim, NLSolversBase
using BenchmarkTools

function fun(x, w)
    mapreduce((xi, wi) -> xi * wi, +, x, w)
end

function fun_grad!(g, x, w)
    g .= w
end

function fun_hess!(h, x, w)
    h .= 0.0
end

const n = 10
const x0 = zeros(n)
const lx = ones(n) * -1.0
const ux = ones(n)
const rng = MersenneTwister(1234)

function solve_subproblem(w)
    f = (x) -> fun(x, w)
    g! = (g, x) -> fun_grad!(g, x, w)
    h! = (h, x) -> fun_hess!(h, x, w)

    df = TwiceDifferentiable(f, g!, h!, x0)
    dfc = TwiceDifferentiableConstraints(lx, ux)
    
    return optimize(df, dfc, x0, IPNewton())
end

function solve(m)
    for i = 1:m
        w = rand(rng, n)
        solve_subproblem(w)
    end
end

@benchmark solve(10000)

