using StatsFuns
using Distributions

abstract type AState end

struct AStateIsotonic <: AState
    α::Float64
end

function reset(astate::AStateIsotonic)
end

function afunc(l::Int64, obs::StateObservable, astate::AStateIsotonic)
    return astat_isotonic(obs.n, obs.W[l]) > log(astate.α)
end

function astat_isotonic(n::Int64, W::Vector{Int64})
    # @assert all(0 .<= positive_counts .<= n)
    n_visits = length(W)
    pcon = sum(W) / (n * n_visits)
    lcon = sum(logpdf(Binomial(n, pcon), W[i]) for i = 1:n_visits)
    y = 2 * asin.(sqrt.(W ./ n)) # allocates memory
    ir = isotonic_regression!(y)
    # piso = sin.(ir ./ 2).^2 # allocates memory
    liso = sum(logpdf(Binomial(n, sin(ir[i] / 2)^2), W[i]) for i = 1:n_visits)
    return liso - lcon
end

struct AStateLogistic <: AState
    α::Float64
    βu::Float64
    zu::Float64
end

function reset(astate::AStateLogistic)
end

function afunc(l::Int64, obs::StateObservable, astate::AStateLogistic)
    return astat_logistic(obs.t[l], obs.W[l], obs.n, astate.βu, astate.zu) > log(astate.α)
end

function astat_logistic(t::AbstractVector{Int64}, W::AbstractVector{Int64}, n, βu, zu)
    n_visits = length(W)
    pcon = sum(W) / (n * n_visits)
    lcon = sum(logpdf(Binomial(n, pcon), W[i]) for i = 1:n_visits)
    tp = t[end] # must be end time
    Wp = W[end]
    ts = @view(t[1:end-1])
    Ws = @view(W[1:end-1])
    _, β, z, Γ = solve_logistic(tp, Wp, ts, Ws, n, βu, zu)
    llogistic = normalized_log_likelihood(β, z, Γ, tp, Wp, t, W, n)
    return llogistic - lcon
end