using Random, Distributions
using StatsBase, StatsFuns
using Optim, NLSolversBase, LineSearches, PositiveFactorizations
# import Convex, Mosek, MosekTools
using StaticArrays
using FastClosures
using LinearAlgebra
using Plots

### Logistic Growth Model Subproblem Solver
function activeset(x0, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    βu::Float64, zu::Float64; maxiters = 1000)

    # Free, Free
    x, g = newtonβz(x0, Γ, tp, Wp, t, W, n)
    !is_kkt(x, g, βu, zu) || return x

    # Free, Upper
    x0c = @SVector [x0[1], zu]
    x, g = newtonβ(x0c, Γ, tp, Wp, t, W, n)
    !is_kkt(x, g, βu, zu) || return x

    # Lower, Free
    x0c = @SVector [0.0, x0[2]]
    x, g = newtonz(x0c, Γ, tp, Wp, t, W, n)
    !is_kkt(x, g, βu, zu) || return x

    # Lower, Upper
    x = @SVector [0.0, zu]
    g = log_likelihood_grad(x, Γ, tp, Wp, t, W, n)
    !is_kkt(x, g, βu, zu) || return x

    # Upper, Free
    x0c = @SVector [βu, x0[2]]
    x, g = newtonz(x0c, Γ, tp, Wp, t, W, n)
    !is_kkt(x, g, βu, zu) || return x
    
    return @SVector [βu, zu]
end

function is_kkt(x, g, βu, zu)
    is_primal_feasible(x, βu, zu) && is_dual_feasible(x, g, βu, zu)
end

function is_primal_feasible(x, βu, zu) 
    (0.0 <= x[1] <= βu) && (x[2] <= zu)
end

function is_dual_feasible(x, g, βu, zu)
    all(lagrange_multipliers(x, g, βu, zu) .>= 0.0)
end

function lagrange_multipliers(x, g, βu, zu)
    λ_βl = x[1] > 0.0 ? 0.0 : g[1]
    λ_βu = x[2] < βu ? 0.0 : -g[1]
    λ_zu = x[2] < zu ? 0.0 : -g[2]
    return @SVector [λ_βl, λ_βu, λ_zu]
end

function newtonβ(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 100, α0=1.0, hesstol=1e-6)

    for i = 1:maxiters
        # Convergence Test
        g = log_likelihood_grad(x, Γ, tp, Wp, t, W, n)[1]
        !convergence_test(x[1], g) || break

        # Search Direction
        h = log_likelihood_hess(x, Γ, tp, t, n)[1, 1]
        if abs(h) < hesstol # Adjustment
            h = h + 1.0
        end
        s = h \ g

        # Line Search
        ϕ = @closure (α) -> log_likelihood(SVector{2,Float64}(x[1] - α * s, x[2]), Γ, tp, Wp, t, W, n)
        dϕ = @closure (α) -> s * -log_likelihood_grad(SVector{2,Float64}(x[1] - α * s, x[2]), Γ, tp, Wp, t, W, n)[1]
        ϕdϕ = @closure (α) -> (ϕ(α), dϕ(α))        
        α, _ = BackTracking()(ϕ, dϕ, ϕdϕ, α0, ϕ(0.0), dϕ(0.0))

        x = @SVector [x[1] - α * s, x[2]]
    end
    return x, log_likelihood_grad(x, Γ, tp, Wp, t, W, n)
end

function newtonz(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 100, α0=1.0, hesstol=1e-6)

    for i = 1:maxiters
        # Convergence Test
        g = log_likelihood_grad(x, Γ, tp, Wp, t, W, n)[2]
        !convergence_test(x[2], g) || break

        # Search Direction
        h = log_likelihood_hess(x, Γ, tp, t, n)[2, 2]
        if abs(h) < hesstol # Adjustment
            h = h + 1.0
        end
        s = h \ g

        # Line Search
        ϕ = @closure (α) -> log_likelihood(SVector{2,Float64}(x[1], x[2] - α * s), Γ, tp, Wp, t, W, n)
        dϕ = @closure (α) -> s * -log_likelihood_grad(SVector{2,Float64}(x[1], x[2] - α * s), Γ, tp, Wp, t, W, n)[2]
        ϕdϕ = @closure (α) -> (ϕ(α), dϕ(α))        
        α, _ = BackTracking()(ϕ, dϕ, ϕdϕ, α0, ϕ(0.0), dϕ(0.0))

        x = @SVector [x[1], x[2] - α * s]
    end
    return x, log_likelihood_grad(x, Γ, tp, Wp, t, W, n)
end

function newtonβz(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 100, α0=1.0)

    for i = 1:maxiters
        # Convergence Test
        g = log_likelihood_grad(x, Γ, tp, Wp, t, W, n)
        !convergence_test(x, g) || break
        
        # Search Direction
        H = log_likelihood_hess(x, Γ, tp, t, n)
        s = modified_hessian_inv(H) * g

        # Line Search
        ϕ = @closure (α) -> log_likelihood(x - α * s, Γ, tp, Wp, t, W, n)
        dϕ = @closure (α) -> dot(s, -log_likelihood_grad(x - α * s, Γ, tp, Wp, t, W, n))
        ϕdϕ = @closure (α) -> (ϕ(α), dϕ(α))        
        α, _ = BackTracking()(ϕ, dϕ, ϕdϕ, α0, ϕ(0.0), dϕ(0.0))

        # Step
        x = x - α * s
    end
    return x, log_likelihood_grad(x, Γ, tp, Wp, t, W, n)
end

function modified_hessian_inv(H, tol = sqrt(eps()))
    d, V = eigen(Symmetric(H))
    d = abs.(d)
    d = d + 1.0 * (d .< tol)
    Dinv = @SMatrix [1 / d[1] 0; 0 1 / d[2]]
    return V * Dinv * V'
end

function convergence_test(x, g, tol=1e-3)
    return maximum(abs.(g)) <= tol
end

function solve_logistic_Γ_subproblem(Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    βu::Float64, zu::Float64, β0::Float64 = βu/10, z0::Float64 = zu - 1.0)

    @assert(βu > 0.0)

    if Γ >= tp # so all tΓ are 0, just use standard MLE with constraint
        β = 0.0 # unindentifiable, assume a default value
        z = min(logit((sum(W) + Wp) / (n * (length(W) + 1))), zu)
        x = @SVector [β, z]
        obj = -log_likelihood(x, Γ, tp, Wp, t, W, n)
        return obj, β, z
    end

    x0 = @SVector [β0, z0]
    x = activeset(x0, Γ, tp, Wp, t, W, n, βu, zu)
    obj = -log_likelihood(x, Γ, tp, Wp, t, W, n)

    return obj, x[1], x[2]
end

### Logistic Growth Model Convex.jl Subproblem Solver
function solve_logistic_Γ_subproblem_convex(Γ, t, W, n, βu, zu)
    tΓ = max.(0, t .- Γ)
    β = Convex.Variable(Convex.Positive())
    z = Convex.Variable()
    coeff = β * tΓ + z
    obj = Convex.dot(W, coeff) - n * Convex.logisticloss(coeff)
    problem = Convex.maximize(obj, β <= βu, z <= zu)
    Convex.solve!(problem, () -> Mosek.Optimizer(QUIET=true), verbose=false)
    return problem.optval, Convex.evaluate(β), Convex.evaluate(z)
end

### Logistic Growth Model Profile Likelihood Solver
function solve_logistic(tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    βu::Float64, zu::Float64, β0::Float64 = βu/10, z0::Float64 = zu - 1.0)
    max_obj = -Inf64
    βs = 0.0
    zs = 0.0
    Γs = 0
    for Γ = 1:tp
        obj, β, z = solve_logistic_Γ_subproblem(Γ, tp, Wp, t, W, n, βu, zu, β0, z0)
        β0, z0 = β, z
        if obj >= max_obj
            max_obj = obj
            βs, zs, Γs = β, z, Γ
        end
    end
    return max_obj, βs, zs, Γs
end

function solve_logistic_mt(tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    βu::Float64, zu::Float64, β0::Float64 = βu/10, z0::Float64 = zu - 1.0)
    # This is a multi-threaded version of solve_logistic 
    β0_warm = Threads.Atomic{Float64}(β0)
    z0_warm = Threads.Atomic{Float64}(z0)
    objs = fill(-Inf, Threads.nthreads())
    βs = fill(0.0, Threads.nthreads())
    zs = fill(0.0, Threads.nthreads())
    Γs = fill(0, Threads.nthreads())
    Threads.@threads for Γ = 1:tp
        obj, β, z = solve_logistic_Γ_subproblem(Γ, tp, Wp, t, W, n, βu, zu, β0_warm[], z0_warm[])
        Threads.atomic_xchg!(β0_warm, β)
        Threads.atomic_xchg!(z0_warm, z)
        if obj > objs[threadid()]
            objs[threadid()] = obj
            βs[threadid()] = β
            zs[threadid()] = z
            Γs[threadid()] = Γ
        end
    end
    max_obj, i = findmax(objs)
    return max_obj, βs[i], zs[i], Γs[i]
end

function profile_log_likelihood(tp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    βu::Float64, zu::Float64, β0::Float64 = βu/10, z0::Float64 = zu - 1.0)
    # @assert(n > 0)
    # @assert all(t .>= 1) # time starts at 1
    # @assert issorted(t)
    # @assert tp > t[end]
    # @assert length(t) == length(W)
    # @assert all(0 .<= W .<= n)
    lp = zeros(n + 1)
    Threads.@threads for i = 0:n
        _, β, z, Γ = solve_logistic(tp, i, t, W, n, βu, zu, β0, z0)
        β0, z0 = β, z # warm start
        lp[i+1] = normalized_log_likelihood(β, z, Γ, tp, i, t, W, n)
    end
    return lp
end

function profile_likelihood(tp, t, W, n, βu, zu)
    lp = profile_log_likelihood(tp, t, W, n, βu, zu)
    softmax!(lp)
    return lp
end

function plot_profile_likelihood(tp, t, W, n, βu, zu; path = "")
    pl = profile_likelihood(tp, t, W, n, βu, zu)
    bar(0:n, pl, xlabel = "Number of Positive Tests", ylabel = "Probability", 
        legend=false, title = "Profile Likelihood for time $(tp) at time $(Int(maximum(t)))")
    savefig(joinpath(path, "profile_likelihood_$(tp)_$(Int(maximum(t))).pdf"))
    return pl
end

### Logistic Growth Model Equations
function f_coeff(β, z, Γ::Int64, t::Int64)
    tΓ = max(0, t - Γ)
    return tΓ, β * tΓ  + z
end

function logistic_prevalance(β::Float64, z::Float64, Γ::Int64, t::Int64)
    _, coeff = f_coeff(β, z, Γ, t)
    return logistic(coeff)
end

function normalized_log_likelihood_scalar(β::Float64, z::Float64, Γ::Int64, t::Int64, W::Int64, n::Int64)
    p = logistic_prevalance(β, z, Γ, t)
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

function log_likelihood(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    β, z = x[1], x[2]
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
    g1 = (-W + n * sigd1) * tΓ
    g2 =  -W + n * sigd1
    return @SVector [g1, g2]
end

function log_likelihood_grad(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    β, z = x[1], x[2]
    g = @SVector zeros(2)
    for i = 1:length(W)
        g = g + log_likelihood_grad_scalar(β, z, Γ, t[i], W[i], n)
    end
    g = g + log_likelihood_grad_scalar(β, z, Γ, tp, Wp, n)
    return g
end

function log_likelihood_hess_scalar(β::Float64, z::Float64, Γ::Int64, t::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd2 = logistic(coeff) * logistic(-coeff)
    h11 = n * sigd2 * tΓ^2
    h12 = n * sigd2 * tΓ
    h22 = n * sigd2
    return @SMatrix [h11 h12; h12 h22]
end

function log_likelihood_hess(x, Γ::Int64, tp::Int64, t::AbstractVector{Int64}, n::Int64)
    β, z = x[1], x[2]
    H = @SMatrix zeros(2, 2)
    for i = 1:length(t)
        H = H + log_likelihood_hess_scalar(β, z, Γ, t[i], n)
    end
    H = H + log_likelihood_hess_scalar(β, z, Γ, tp, n)
    return H
end