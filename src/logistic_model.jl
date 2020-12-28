using Random, Distributions
using StatsBase, StatsFuns
using Optim, NLSolversBase, LineSearches
import Convex, Mosek, MosekTools
using Plots

# Initial values, lower and upper bounds for beta and z
const x0 = [0.01, logit(0.01)]
const lx = [-1e6, -1e6]
const ux = [1e6, 1e6]
const linesearch = BackTracking()

### Projected Gradient Descent
function pgd(β, z, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 1000, α0 = 1.0)
    gβ, gz = log_likelihood_grad(β, z, Γ, tp, Wp, t, W, n)
    ϕ1 = (α) -> ϕ(α, β, z, gβ, gz, Γ, tp, Wp, t, W, n)
    # dϕ1 = (α) -> dϕ(α, β, z, gβ, gz, Γ, tp, Wp, t, W, n)
    # println(β)
    i = 1
    # 
    while (i <= maxiters) #&& !(convergence_test(β, lx[1], ux[1], gβ) && convergence_test(z, lx[2], ux[2], gz))
        # ϕdϕ1 = (α) -> ϕdϕ(α, β, z, gβ, gz, Γ, tp, Wp, t, W, n)    
        # ϕ0, dϕ0 = ϕdϕ1(0.0)
        # α, _ = linesearch(ϕ1, dϕ1, ϕdϕ1, α0, ϕ0, dϕ0)
        # α = 1.0
        # β, z = β - α * gβ, z - α * gz
        β = β - 1.0
        # gβ, gz = log_likelihood_grad(β, z, Γ, tp, Wp, t, W, n)
        i += 1
    end
    return β, z
end

function ϕ(α::Float64, β::Float64, z::Float64, gβ::Float64, gz::Float64, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    return log_likelihood(β - α * gβ, z - α * gz, Γ, tp, Wp, t, W, n)
end

function dϕ(α, β, z, gβ, gz, Γ, tp, Wp, t, W, n)
    gβα, gzα = log_likelihood_grad(β - α * gβ, z - α * gz, Γ, tp, Wp, t, W, n)
    return gβα * -gβ + gzα * -gz
end

function ϕdϕ(α, β, z, gβ, gz, Γ, tp, Wp, t, W, n)
    return ϕ(α, β, z, gβ, gz, Γ, tp, Wp, t, W, n), dϕ(α, β, z, gβ, gz, Γ, tp, Wp, t, W, n)
end

function convergence_test(x, l, u, g, tol=1e-3) # returns true if converged
    return (abs(g) < tol) #|| ((x == l) && (-g < 0.0)) || ((x == u) && (-g > 0.0))
end

function box_projection(x, l, u)
    if x > u
        return u
    elseif x < l
        return l
    end
    return x
end

### Optim
function f_coeff(β, z, Γ::Int64, t::Int64)
    tΓ = max(0, t - Γ)
    return tΓ, β * tΓ  + z
end

function logistic_prevalance(β::Float64, z::Float64, Γ::Int64, tp::Int64)
    _, coeff = f_coeff(β, z, Γ, tp)
    return logistic(coeff)
end

function normalized_log_likelihood_scalar(β::Float64, z::Float64, Γ::Int64, t::Int64, W::Int64, n::Int64)
    _, coeff = f_coeff(β, z, Γ, t)
    p = logistic(coeff)
    return logpdf(Binomial(n, p), W)
end

function normalized_log_likelihood(β::Float64, z::Float64, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
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

function log_likelihood(β, z, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    f = 0.0
    for i = 1:length(W)
        f -= log_likelihood_scalar(β, z, Γ, t[i], W[i], n)
    end
    f -= log_likelihood_scalar(β, z, Γ, tp, Wp, n)
    return f
end

function log_likelihood_grad_scalar(β::Float64, z::Float64, Γ::Int64, t::Int64, W::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd1 = logistic(coeff)
    return W * tΓ - n * sigd1 * tΓ, W - n * sigd1
end

function log_likelihood_grad(β, z, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    gβ, gz = 0.0, 0.0
    for i = 1:length(W)
        tmp1, tmp2 = log_likelihood_grad_scalar(β, z, Γ, t[i], W[i], n)
        gβ -= tmp1 
        gz -= tmp2 
    end
    tmp1, tmp2 = log_likelihood_grad_scalar(β, z, Γ, tp, Wp, n)
    gβ -= tmp1 
    gz -= tmp2
    return gβ, gz
end

function log_likelihood_hess_scalar!(h::Array{Float64}, β::Float64, z::Float64, Γ::Int64, t::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd2 = logistic(coeff) * (1 - logistic(coeff))
    h[1, 1] -= -n * (sigd2 * tΓ^2)
    h[1, 2] -= -n * tΓ * sigd2
    h[2, 1] -= -n * tΓ * sigd2
    h[2, 2] -= -n * sigd2
end

function log_likelihood_hess!(h::Array{Float64}, x::Vector{Float64}, Γ::Int64, tp::Int64, t::AbstractVector{Int64}, n::Int64)
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

function solve_logistic_Γ_subproblem_optim(Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    # fun = (x) -> log_likelihood(x, Γ, tp, Wp, t, W, n)
    # fun_grad! = (g, x) -> log_likelihood_grad!(g, x, Γ, tp, Wp, t, W, n)
    # fun_hess! = (h, x) -> log_likelihood_hess!(h, x, Γ, tp, t, n)
    
    # df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    # dfc = TwiceDifferentiableConstraints(lx, ux)
    
    # res = optimize(df, dfc, x0, IPNewton(), Optim.Options(iterations = 10))
    # obj::Float64 = -Optim.minimum(res)
    # β::Float64, z::Float64 = Optim.minimizer(res)
    β, z = pgd(0.01, logit(0.01), Γ, tp, Wp, t, W, n)
    obj = -log_likelihood(β, z, Γ, tp, Wp, t, W, n)


    return obj, β, z
end

function solve_logistic_optim(tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
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

function profile_log_likelihood(tp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    @assert tp > maximum(t)
    @assert all(0 .<= W .<= n)
    @assert all(t .>= 0)
    lp = zeros(n + 1)
    for i = 0:n #Threads.@threads 
        _, β, z, Γ = solve_logistic_optim(tp, i, t, W, n)
        lp[i+1] = normalized_log_likelihood(β, z, Γ, tp, i, t, W, n)
    end
    return lp
end

function profile_likelihood(tp, t, W, n)
    return softmax(profile_log_likelihood(tp, t, W, n))
end

function plot_profile_likelihood(tp, t, W, n; path = "")
    pl = profile_likelihood(tp, t, W, n)
    bar(0:n, pl, xlabel = "Number of Positive Tests", ylabel = "Probability", 
        legend=false, title = "Profile Likelihood for time $(tp) at time $(Int(maximum(t)))")
    savefig(joinpath(path, "profile_likelihood_$(tp)_$(Int(maximum(t))).pdf"))
    return pl
end

# ### Convex.jl
function solve_logistic_Γ_subproblem_convex(Γ, t, W, n)
    tΓ = max.(0, t .- Γ)
    β = Convex.Variable(Convex.Positive())
    z = Convex.Variable()
    coeff = β * tΓ + z
    obj = Convex.dot(W, coeff) - n * Convex.logisticloss(coeff)
    problem = Convex.maximize(obj, β >= lx[1], z >= lx[2], β <= ux[1], z <= ux[2])
    Convex.solve!(problem, () -> Mosek.Optimizer(QUIET=true), verbose=false)
    return problem.optval, Convex.evaluate(β), Convex.evaluate(z)
end