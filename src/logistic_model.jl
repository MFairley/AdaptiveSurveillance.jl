using Random, Distributions
using StatsBase, StatsFuns
using Optim, NLSolversBase, LineSearches, PositiveFactorizations
import Convex, Mosek, MosekTools
using StaticArrays
using FastClosures
using LinearAlgebra
using Plots

const lx = [0.0, -Inf] # Lower bound does not affect Convex.jl version
const ux = [0.1, logit(0.1)]

# options:
# 1) projected newtons method - need further references to understand
# 2) active set method - easy to understand and robust but annoying to code
# 3) interior point - complex to implement, numerical issues

function activeset(x0, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 1000)

    # free, free -> primal feasible? dual feasible?
    x, g = newtonβz(x0, Γ, tp, Wp, t, W, n)
    # println("1")
    # println(x)
    # println(g)
    !is_kkt(x, g) || return x

    # free, u
    x, g = newtonβ([x0[1], ux[2]], Γ, tp, Wp, t, W, n)
    # println("2")
    # println(g)
    !is_kkt(x, g) || return x

    # l, free
    x, g = newtonz([0.0, x0[2]], Γ, tp, Wp, t, W, n)
    # println("3")
    !is_kkt(x, g) || return x

    # l, u <- fixed
    x = [0.0, ux[2]]
    g = zeros(2)
    log_likelihood_grad!(g, x, Γ, tp, Wp, t, W, n)
    # println("4")
    !is_kkt(x, g) || return x

    # u, free
    x, g = newtonz([ux[1], x0[2]], Γ, tp, Wp, t, W, n)
    println("5")
    !is_kkt(x, g) || return x
    
    # u, u -< only option left
    println("6")
    return ux
end

function is_kkt(x, g)
    is_primal_feasible(x) && is_dual_feasible(x, g)
end

function is_primal_feasible(x) # add a tolerance? 
    (0.0 <= x[1] <= ux[1]) && (x[2] <= ux[2])
end

function is_dual_feasible(x, g)
    all(lagrange_multipliers(x, g) .>= 0.0) # add a tolerance?
end

function lagrange_multipliers(x, g)
    λ_βl = x[1] > 0.0 ? 0.0 : g[1]
    λ_βu = x[2] < ux[1] ? 0.0 : -g[1]
    # @assert (λ_βl == 0.0) || (λ_βu == 0.0)
    λ_zu = x[2] < ux[2] ? 0.0 : -g[2]
    return λ_βl, λ_βu, λ_zu
end

# function ipnewton(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
#     maxiters = 1000)
#     # interior point newton to do
#     # see https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/solvers/constrained/ipnewton/ipnewton.jl

# end

function newtonβ(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 1000, α0=1.0)

    g = zeros(2)
    H = zeros(2, 2)
    for i = 1:maxiters
        log_likelihood_grad!(g, x, Γ, tp, Wp, t, W, n)
        log_likelihood_hess!(H, x, Γ, tp, t, n)
        if abs(H[1, 1]) < 1e-6
            H[1, 1] += 1
        end
        s = H[1, 1] \ g[1]

        ϕ = @closure (α) -> log_likelihood([x[1] - α * s, x[2]], Γ, tp, Wp, t, W, n)
        function dϕ(α) 
            gα = zeros(2)
            log_likelihood_grad!(gα, [x[1] - α * s, x[2]], Γ, tp, Wp, t, W, n)
            s * - gα[1]
        end
        ϕdϕ = @closure (α) -> (ϕ(α), dϕ(α))
        α, _ = BackTracking()(ϕ, dϕ, ϕdϕ, α0, ϕ(0.0), dϕ(0.0))

        x = [x[1] - α * s, x[2]]

        if convergence_test(x[1], g[1], H[1, 1])
            break
        end
    end
    return x, g
end

function newtonz(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 1000, α0=1.0)

    g = zeros(2)
    H = zeros(2, 2)
    for i = 1:maxiters
        log_likelihood_grad!(g, x, Γ, tp, Wp, t, W, n)
        log_likelihood_hess!(H, x, Γ, tp, t, n)
        if abs(H[2, 2]) < 1e-6
            H[2, 2] += 1
        end
        s = H[2, 2] \ g[2] # search direction

        ϕ = @closure (α) -> log_likelihood([x[1], x[2] - α * s], Γ, tp, Wp, t, W, n)
        function dϕ(α) 
            gα = zeros(2)
            log_likelihood_grad!(gα, [x[1], x[2] - α * s], Γ, tp, Wp, t, W, n)
            s * - gα[2]
        end
        ϕdϕ = @closure (α) -> (ϕ(α), dϕ(α))
        α, _ = BackTracking()(ϕ, dϕ, ϕdϕ, α0, ϕ(0.0), dϕ(0.0))

        x = [x[1], x[2] - α * s]

        if convergence_test(x[2], g[2], H[2, 2])
            break
        end
    end
    return x, g
end

function newtonβz(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64;
    maxiters = 10, α0=1.0)

    g = zeros(2) # to do: change this to non-memory allocating
    H = zeros(2, 2)
    # can probably make non-allocating by using SVector from StaticArrays and changing grad and hess to not be in place
    # StaticArrays lets you do usual linear algebra operations otherwise will need to manually implement those oeprations to prevent
    # memory allocation
    # https://github.com/JuliaArrays/StaticArrays.jl
    for i = 1:maxiters
        log_likelihood_grad!(g, x, Γ, tp, Wp, t, W, n)
        log_likelihood_hess!(H, x, Γ, tp, t, n)
        F = PositiveFactorizations.cholesky!(Positive, H) # adjusted hessian to deal with near positive definite matrices
        s = F\g # search direction

        ϕ = @closure (α) -> log_likelihood(x - α * s, Γ, tp, Wp, t, W, n)
        function dϕ(α) 
            gα = zeros(2)
            log_likelihood_grad!(gα, x - α * s, Γ, tp, Wp, t, W, n)
            dot(s, -gα)
        end
        ϕdϕ = @closure (α) -> (ϕ(α), dϕ(α))
        α, _ = BackTracking()(ϕ, dϕ, ϕdϕ, α0, ϕ(0.0), dϕ(0.0))

        x = x - α * s

        if convergence_test(x, g, H) # should do the gradient of the new point here!, otherwise will do an extra iteration
            break
        end
    end
    return x, g
end

function convergence_test(x, g, H, tol=1e-3)
    return maximum(abs.(g)) <= tol
end

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

function log_likelihood(x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    β, z = x[1], x[2]
    f = 0.0
    for i = 1:length(W)
        f -= log_likelihood_scalar(β, z, Γ, t[i], W[i], n)
    end
    f -= log_likelihood_scalar(β, z, Γ, tp, Wp, n)
    return f
end

function log_likelihood_grad_scalar!(g, β::Float64, z::Float64, Γ::Int64, t::Int64, W::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd1 = logistic(coeff)
    g[1] += -W * tΓ + n * sigd1 * tΓ
    g[2] += -W + n * sigd1
end

function log_likelihood_grad!(g, x, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    β, z = x[1], x[2]
    g[1] = 0.0
    g[2] = 0.0
    for i = 1:length(W)
        log_likelihood_grad_scalar!(g, β, z, Γ, t[i], W[i], n)
    end
    log_likelihood_grad_scalar!(g, β, z, Γ, tp, Wp, n)
    # println(g)
end

function log_likelihood_hess_scalar!(h, β::Float64, z::Float64, Γ::Int64, t::Int64, n::Int64)
    tΓ, coeff = f_coeff(β, z, Γ, t)
    sigd2 = logistic(coeff) * logistic(-coeff)
    h[1, 1] += n * sigd2 * tΓ^2
    h[1, 2] += n * sigd2 * tΓ
    # h[2, 1] += n * sigd2 * tΓ
    h[2, 2] += n * sigd2
end

function log_likelihood_hess!(h, x, Γ::Int64, tp::Int64, t::AbstractVector{Int64}, n::Int64)
    β, z = x[1], x[2]
    h[1, 1] = 0.0
    h[1, 2] = 0.0
    h[2, 2] = 0.0
    for i = 1:length(t)
        log_likelihood_hess_scalar!(h, β, z, Γ, t[i], n)
    end
    log_likelihood_hess_scalar!(h, β, z, Γ, tp, n)
    h[2, 1] = h[1, 2]
    # println(h)
end

function log_likelihood_hess_chol(h::Array{Float64}, x::Vector{Float64}, Γ::Int64, tp::Int64, t::AbstractVector{Int64}, n::Int64)
    L = zeros(2, 2)
    L[1, 1] = sqrt(h[1, 1])
    L[2, 1] = h[1, 2] / L[1, 1]
    L[2, 2] = sqrt(h[2, 2] - L[2, 1]^2)
    return L
end

function solve_logistic_Γ_subproblem(β0::Float64, z0::Float64, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    x0 = [0.01, logit(0.01)]

    if Γ >= tp # so all tΓ are 0, just use standard MLE
        β = 0.0 # unindentifiable 
        z = logit((sum(W) + Wp) / (n * (length(W) + 1))) # should set a constraint on this
        obj = -log_likelihood([β, z], Γ, tp, Wp, t, W, n)
        return obj, β, z
    end

    x = activeset(x0, Γ, tp, Wp, t, W, n)
    # x, _ = newtonβz(x0, Γ, tp, Wp, t, W, n)
    β, z = x
    obj = -log_likelihood(x, Γ, tp, Wp, t, W, n)

    return obj, β, z
end

function solve_logistic_Γ_subproblem_optim(β0::Float64, z0::Float64, Γ::Int64, tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64)
    x0 = [0.01, logit(0.01)] # using warm start points fails due to not being in interior
    
    fun = (x) -> log_likelihood(x, Γ, tp, Wp, t, W, n)
    fun_grad! = (g, x) -> log_likelihood_grad!(g, x, Γ, tp, Wp, t, W, n)
    fun_hess! = (h, x) -> log_likelihood_hess!(h, x, Γ, tp, t, n)
    
    if Γ >= tp # so all tΓ are 0, just use standard MLE
        β = 0.0 # unindentifiable 
        z = logit((sum(W) + Wp) / (n * (length(W) + 1))) # should set a constraint on this
        obj = -fun([β, z])
        return obj, β, z
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    dfc = TwiceDifferentiableConstraints(lx, ux)
    
    res = optimize(df, dfc, x0, IPNewton())
    # res = optimize(df, x0, Newton(;linesearch = LineSearches.BackTracking())) # unconstrained, line search sometimes fails
    # res = optimize(df, x0, Newton()) # unconstrained
    obj = -Optim.minimum(res)
    β, z = Optim.minimizer(res)

    return obj, β, z
end

function solve_logistic_optim(tp::Int64, Wp::Int64, t::AbstractVector{Int64}, W::AbstractVector{Int64}, n::Int64,
    β0::Float64 = 0.01, z0::Float64 = 0.0)
    max_obj = -Inf64
    βs = 0.0
    zs = 0.0
    Γs = 0
    for Γ = 0:tp # type instability here with Threads.@threads
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
    Threads.@threads for i = 0:n
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
    problem = Convex.maximize(obj, β <= ux[1], z <= ux[2])
    Convex.solve!(problem, () -> Mosek.Optimizer(QUIET=true), verbose=false)
    return problem.optval, Convex.evaluate(β), Convex.evaluate(z)
end