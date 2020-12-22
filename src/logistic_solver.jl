using Random, Distributions
using StatsBase, StatsFuns
using Optim, NLSolversBase
import Convex, Mosek, MosekTools
using Plots

### Optim
function f_coeff(β, z, Γ::Int64, t::Int64)
    tΓ = max(0, t - Γ)
    return tΓ, β * tΓ  + z
end

function normalized_log_likelihood_scalar(β::Float64, z::Float64, Γ::Int64, t::Int64, W::Int64, n::Int64)
    _, coeff = f_coeff(β, z, Γ, t)
    p = logistic(coeff)
    return logpdf(Binomial(n, p), W)
end

function normalized_log_likelihood(β::Float64, z::Float64, Γ::Int64, tp::Int64, Wp::Int64, t::Vector{Int64}, W::Vector{Int64}, n::Int64)
    f = 0.0
    for i = 1:length(W)
        f += normalized_log_likelihood_scalar(β, z, Γ, t[i], W[i], n)
    end
    f += normalized_log_likelihood_scalar(β, z, Γ, tp, Wp, n)
    return f
end

function log_likelihood_scalar(β, z, Γ::Int64, t::Int64, W::Int64, n::Int64)
    _, coeff = f_coeff(β, z, Γ, t)
    return W * coeff - n * log1pexp(coeff)
end

function log_likelihood(x, Γ::Int64, tp::Int64, Wp::Int64, t::Vector{Int64}, W::Vector{Int64}, n::Int64)
    β, z = x[1], x[2]
    f = 0.0
    for i = 1:length(W)
        f -= log_likelihood_scalar(β, z, Γ, t[i], W[i], n)
    end
    f -= log_likelihood_scalar(β, z, Γ, tp, Wp, n)
    return f
end

function log_likelihood_grad_scalar!(g::Vector{Float64}, β::Float64, z::Float64, Γ::Int64, t::Int64, W::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd1 = logistic(coeff)
    g[1] -= W * tΓ - n * sigd1 * tΓ
    g[2] -= W - n * sigd1
end

function log_likelihood_grad!(g::Vector{Float64}, x::Vector{Float64}, Γ::Int64, tp::Int64, Wp::Int64, t::Vector{Int64}, W::Vector{Int64}, n::Int64)
    β, z = x[1], x[2]
    g[1] = 0.0
    g[2] = 0.0
    for i = 1:length(W)
        log_likelihood_grad_scalar!(g, β, z, Γ, t[i], W[i], n)
    end
    log_likelihood_grad_scalar!(g, β, z, Γ, tp, Wp, n)
end

function log_likelihood_hess_scalar!(h::Array{Float64}, β::Float64, z::Float64, Γ::Int64, t::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd2 = logistic(coeff) * (1 - logistic(coeff))
    h[1, 1] -= -n * (sigd2 * tΓ^2)
    h[1, 2] -= -n * tΓ * sigd2
    h[2, 1] -= -n * tΓ * sigd2
    h[2, 2] -= -n * sigd2
end

function log_likelihood_hess!(h::Array{Float64}, x::Vector{Float64}, Γ::Int64, tp::Int64, t::Vector{Int64}, n::Int64)
    β, z = x[1], x[2]
    h[1, 1] = 0.0
    h[1, 2] = 0.0
    h[2, 1] = 0.0
    h[2, 2] = 0.0
    for i = 1:length(t)
        log_likelihood_hess_scalar!(h, β, z, Γ, t[i], n)
    end
    log_likelihood_hess_scalar!(h, β, z, Γ, tp, n)
end

function solve_logistic_Γ_subproblem_optim(Γ::Int64, tp::Int64, Wp::Int64, t::Vector{Int64}, W::Vector{Int64}, n::Int64;
    x0 = [0.01, logit(0.01)], lx = [0.0, -Inf], ux = [1.0, logit(0.5)])
    fun = (x) -> log_likelihood(x, Γ, tp, Wp, t, W, n)
    fun_grad! = (g, x) -> log_likelihood_grad!(g, x, Γ, tp, Wp, t, W, n)
    fun_hess! = (h, x) -> log_likelihood_hess!(h, x, Γ, tp, t, n)
    
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    dfc = TwiceDifferentiableConstraints(lx, ux)
    
    res = optimize(df, dfc, x0, IPNewton())
    obj::Float64 = -Optim.minimum(res)
    β::Float64, z::Float64 = Optim.minimizer(res)

    return obj, β, z
end

function solve_logistic_optim(tp::Int64, Wp::Int64, t::Vector{Int64}, W::Vector{Int64}, n::Int64)
    max_obj = -Inf64
    βs = 0.0
    zs = 0.0
    Γs = 0
    for Γ = 0:maximum(t) # type instability here with Threads.@threads
        obj, β, z = solve_logistic_Γ_subproblem_optim(Γ, tp, Wp, t, W, n)
        if obj >= max_obj
            max_obj = obj
            βs, zs, Γs = β, z, Γ
        end
    end
    return max_obj, βs, zs, Γs
end

function profile_log_likelihood(n1::Int64, n2::Int64, tp::Int64, t::Vector{Int64}, W::Vector{Int64}, n::Int64)
    @assert n1 <= n2
    @assert tp > maximum(t)
    @assert 0 <= n1 <= n
    @assert 0 <= n2 <= n
    @assert all(0 .<= W .<= n)
    @assert all(t .>= 0)
    lp = zeros(n2 - n1 + 1)
    Wp_range = n1:n2
    Threads.@threads for i = 1:length(Wp_range)
        _, β, z, Γ = solve_logistic_optim(tp, Wp_range[i], t, W, n)
        lp[i] = normalized_log_likelihood(β, z, Γ, tp, Wp_range[i], t, W, n)
    end
    return lp
end

function future_alarm_log_probability(n1, n2, tp, t, W, n)
    return logsumexp(profile_log_likelihood(n1, n2, tp, t, W, n))
end

function profile_likelihood(tp, t, W, n)
    return softmax(profile_log_likelihood(0, n, tp, t, W, n))
end

function plot_profile_likelihood(tp, t, W, n; path = "")
    pl = profile_likelihood(tp, t, W, n)
    bar(0:n, pl, xlabel = "Number of Positive Tests", ylabel = "Probability", 
        legend=false, title = "Profile Likelihood for time $(tp) at time $(Int(maximum(t)))")
    savefig(joinpath(path, "profile_likelihood_$(tp)_$(Int(maximum(t))).pdf"))
    return pl
end

# ### Convex.jl
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