using Optim, NLSolversBase, Random, Distributions
using StatsBase
import Convex, Mosek, MosekTools
using Plots
# using LinearAlgebra

### Optim
function normalized_log_likelihood(β::Float64, z::Float64, Γ::Int64, t::Array{Int64}, W::Array{Float64}, n::Int64)
    tΓ = max.(0, t .- Γ)
    p = logistic.(β * tΓ .+ z)
    return sum(logpdf(Binomial(n, p[i]), W[i]) for i = 1:length(W))
end

function log_likelihood(x, tΓ::Array{Int64}, W::Array{Float64}, n::Int64)
    β, z = x[1], x[2]
    f = 0.0
    for i = 1:length(W)
        coeff = β * tΓ[i] + z
        f -= W[i] * coeff - n * log1pexp(coeff)
    end
    return f
end

function log_likelihood_grad!(g::Array{Float64}, x::Array{Float64}, tΓ::Array{Int64}, W::Array{Float64}, n::Int64)
    β, z = x[1], x[2]
    g[1] = 0.0
    g[2] = 0.0
    for i = 1:length(W)
        coeff = β * tΓ[i] + z
        sigd1 = logistic(coeff)
        g[1] -= W[i] * tΓ[i] - n * sigd1 * tΓ[i]
        g[2] -= W[i] - n * sigd1
    end
end

function log_likelihood_hess!(h::Array{Float64}, x::Array{Float64}, tΓ::Array{Int64}, n::Int64)
    β, z = x[1], x[2]
    h[1, 1] = 0.0
    h[1, 2] = 0.0
    h[2, 1] = 0.0
    h[2, 2] = 0.0
    for i = 1:length(tΓ)
        coeff = β * tΓ[i] + z
        sigd2 = logistic(coeff) * (1 - logistic(coeff))
        h[1, 1] -= -n * (sigd2 * tΓ[i]^2)
        h[1, 2] -= -n * tΓ[i] * sigd2
        h[2, 1] -= -n * tΓ[i] * sigd2
        h[2, 2] -= -n * sigd2
    end
end

function solve_logistic_Γ_subproblem_optim(Γ::Int64, t::Array{Int64}, W::Array{Float64}, n::Int64, x0 = [0.01, logit(0.01)], ux = [1.0, logit(0.5)])
    tΓ = max.(0, t .- Γ)
    fun = (x) -> log_likelihood(x, tΓ, W, n)
    fun_grad! = (g, x) -> log_likelihood_grad!(g, x, tΓ, W, n)
    fun_hess! = (h, x) -> log_likelihood_hess!(h, x, tΓ, n)
    
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    dfc = TwiceDifferentiableConstraints([0.0, -Inf], ux)
    res = optimize(df, dfc, x0, IPNewton())
    obj = -Optim.minimum(res)
    β, z = Optim.minimizer(res)

    return obj, β, z
end

function solve_logistic_optim(t::Array{Int64}, W::Array{Float64}, n::Int64)
    max_obj, βs, zs, Γs = -Inf, 0.0, 0.0, 0
    for Γ = 0:maximum(t) # Threads.@threads # to do, fix type instability here 
        obj, β, z = solve_logistic_Γ_subproblem_optim(Γ, t, W, n)
        # obj, β, z = solve_logistic_Γ_subproblem_convex(Γ, t, W, n)
        if obj >= max_obj
            max_obj = obj
            βs, zs, Γs = β, z, Γ
        end
    end
    return max_obj, βs, zs, Γs
end

function profile_log_likelihood(n1::Int64, n2::Int64, tp::Int64, t::Array{Int64}, W::Array{Float64}, n::Int64)
    @assert n1 <= n2
    @assert tp > maximum(t)
    W = vcat(W, n1)
    t = vcat(t, tp)
    lp = zeros(n2 - n1 + 1)
    for (j, i) in enumerate(n1:n2)
        W[end] = i # this makes this not parallel
        _, β, z, Γ = solve_logistic_optim(t, W, n)
        lp[j] = normalized_log_likelihood(β, z, Γ, t, W, n)
    end
    return lp
end

function future_alarm_log_probability(n1, n2, tp, W, t, n)
    return logsumexp(profile_log_likelihood(n1, n2, tp, t, W, n))
end

function profile_likelihood(tp, t, W, n)
    return softmax(profile_log_likelihood(0, n, tp, t, W, n))
end

function plot_profile_likelihood(tp, t, W, n; path = "")
    pl = profile_likelihood(tp, t, W, n)
    bar(n1:n2, pl, xlabel = "Number of Positive Tests", ylabel = "Probability", 
        legend=false, title = "Profile Likelihood for time $(tp) at time $(Int(maximum(t)))")
    savefig(joinpath(path, "profile_likelihood_$(tp)_$(Int(maximum(t))).pdf"))
    return pl
end

### Convex.jl
function solve_logistic_Γ_subproblem_convex(Γ, t, W, n, ux = [1.0, logit(0.5)])
    tΓ = max.(0, t .- Γ)
    β = Convex.Variable(Convex.Positive())
    z = Convex.Variable()
    coeff = β * tΓ + z
    obj = Convex.dot(W, coeff) - n * Convex.logisticloss(coeff)
    problem = Convex.maximize(obj, β <= ux[1], z <= ux[2])
    Convex.solve!(problem, () -> Mosek.Optimizer(QUIET=true), verbose=false)
    return problem.optval, Convex.evaluate(β), Convex.evaluate(z)
end