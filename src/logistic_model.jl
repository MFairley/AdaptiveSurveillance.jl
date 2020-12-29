using Random, Distributions
using StatsBase, StatsFuns
using Optim, NLSolversBase, LineSearches
import Convex, Mosek, MosekTools
using Plots
using FastClosures
using PositiveFactorizations

# Initial values, lower and upper bounds for beta and z
const lx = [-1e6, -1e6]
const ux = [1e6, 1e6]

### Projected Newton's Method
function pgd(β::Float64, z::Float64, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 10, α0 = 1.0)
    
    α = α0 # change to line search later
    for i = 1:maxiters
        gβ, gz = log_likelihood_grad(β, z, Γ, tp, Wp, t, W, n)
        H = log_likelihood_hess(β, z, Γ, tp, t, n)
        tβz = PositiveFactorizations.cholesky(Positive, H)\[ gβ, gz ]
        # tβ = invh11 * gβ + invh12 * gz
        # tz = invh21 * gβ + invh22 * gz
        # println(tβ)
        β, z = β - α * tβz[1], z - α * tβz[2]

        if convergence_test(β, z, gβ, gz)
            break
        end
    end
    return β, z
end

function convergence_test(β, z, gβ, gz, tol=1e-3)
    return (abs(gβ) < tol) && (abs(gz) < tol)
end

# function convergence_test(x, l, u, g, tol=1e-3) # returns true if converged
    # return (abs(g) < tol) #|| ((x == l) && (-g < 0.0)) || ((x == u) && (-g > 0.0))
# end

# function box_projection(x, l, u)
#     if x > u
#         return u
#     elseif x < l
#         return l
#     end
#     return x
# end

### Optim
function f_coeff(β, z, Γ::Int64, t::Int64)
    tΓ = max(0, t - Γ)
    return tΓ, β * tΓ + z
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
    return -W * tΓ + n * sigd1 * tΓ, -W + n * sigd1
end

function log_likelihood_grad(β, z, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    gβ, gz = 0.0, 0.0
    for i = 1:length(W)
        igβ, igz = log_likelihood_grad_scalar(β, z, Γ, t[i], W[i], n)
        gβ, gz = gβ + igβ, gz + igz
    end
    igβ, igz = log_likelihood_grad_scalar(β, z, Γ, tp, Wp, n)
    gβ, gz = gβ + igβ, gz + igz
    return gβ, gz
end

function log_likelihood_hess_scalar(β::Float64, z::Float64, Γ::Int64, t::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd2 = logistic(coeff) * (1 - logistic(coeff))
    h11 = n * sigd2 * tΓ^2
    h12 = n * tΓ * sigd2
    h21 = n * tΓ * sigd2
    h22 = n * sigd2
    return h11, h12, h21, h22
end

function log_likelihood_hess(β::Float64, z::Float64, Γ::Int64, tp::Int64, t::AbstractVector{Int64}, n::Int64)
    # println("β = $β, z = $z, Γ = $Γ")
    # this function is numerically unstable
    h11, h12, h21, h22 = 0.0, 0.0, 0.0, 0.0
    for i = 1:length(t)
        ih11, ih12, ih21, ih22 = log_likelihood_hess_scalar(β, z, Γ, t[i], n) 
        h11, h12, h21, h22 = h11 + ih11, h12 + ih12, h21 + ih21, h22 + ih22
    end
    ih11, ih12, ih21, ih22 = log_likelihood_hess_scalar(β, z, Γ, tp, n)
    # println("ih11 = $ih11 , ih12 = $ih12 , ih21 = $ih21 , ih22 = $ih22")
    tΓ, coeff = f_coeff(β, z, Γ, tp)
    # println(tΓ)
    # println(logistic(coeff)) # issue here since this evaluates to 1.0 exactly, numerical issue
    h11, h12, h21, h22 = h11 + ih11, h12 + ih12, h21 + ih21, h22 + ih22

    invdet = 1 / (h11 * h22 - h21 * h12)

    # println("h11 = $h11 , h12 = $h12 , h21 = $h21 , h22 = $h22")
    invh11, invh12, invh21, invh22 = invdet * h22, -invdet * h21, -invdet * h12, invdet * h11

    return [h11 h12; h21 h22]
end

function solve_logistic_Γ_subproblem_optim(β0::Float64, z0::Float64, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    # if Γ > tp # could skip optimization here and go straight to mle
    β, z = pgd(β0, z0, Γ, tp, Wp, t, W, n)
    obj = -log_likelihood(β, z, Γ, tp, Wp, t, W, n)
    return obj, β, z
end

function solve_logistic_optim(tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    β0::Float64 = 0.01, z0::Float64 = 0.0)
    max_obj = -Inf64
    βs = 0.0
    zs = 0.0
    Γs = 0
    for Γ = 0:maximum(t) # type instability here with Threads.@threads
        obj, β, z = solve_logistic_Γ_subproblem_optim(β0, z0, Γ, tp, Wp, t, W, n)
        β0, z0 = β, z
        if obj >= max_obj
            max_obj = obj
            βs, zs, Γs = β, z, Γ
        end
    end
    return max_obj, βs, zs, Γs
end

function profile_log_likelihood(tp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    β0::Float64 = 0.01, z0::Float64 = 0.0)
    @assert tp > maximum(t)
    @assert all(0 .<= W .<= n)
    @assert all(t .>= 0)
    lp = zeros(n + 1)
    Threads.@threads for i = 0:n #
        _, β, z, Γ = solve_logistic_optim(tp, i, t, W, n, β0, z0)
        β0, z0 = β, z
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