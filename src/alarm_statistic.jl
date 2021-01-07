using StatsFuns
using Distributions

abstract type AState end

struct AStateIsotonic <: AState
    α::Float64
end

function reset(astate::AStateIsotonic)
end

function afunc(t::Int64, obs::StateObservable, astate::AStateIsotonic)
    return astat_isotonic(obs.n, @view(obs.W[obs.x .== obs.x[t]])) > log(astate.α)
end

function astat_isotonic(n::Int64, W::AbstractVector{Int64})
    # @assert all(0 .<= positive_counts .<= n)
    n_visits = length(W)
    pcon = sum(W) / (n * n_visits)
    lcon = sum(logpdf(Binomial(n, pcon), W[i]) for i = 1:n_visits)
    y = 2 * asin.(sqrt.(W ./ n))
    ir = isotonic_regression!(y)
    piso = sin.(ir ./ 2).^2
    liso = sum(logpdf(Binomial(n, piso[i]), W[i]) for i = 1:n_visits)
    return liso - lcon
end


struct AStateLogistic <: AState
    α::Float64
    βu::Float64
    zu::Float64
end

function reset(astate::AStateLogistic)
end

function afunc(t::Int64, obs::StateObservable, astate::AStateLogistic)
    l = obs.x[t]
    past_times = @view((1:obs.maxiters)[obs.x .== l])
    past_counts = @view(obs.W[obs.x .== l])
    return astat_logistic(past_times, past_counts, obs.n, astate.βu, astate.zu) > log(astate.α)
end

function astat_logistic(t::AbstractVector{Int64}, W::AbstractVector{Int64}, n, βu, zu)
    n_visits = length(W)
    pcon = sum(W) / (n * n_visits)
    lcon = sum(logpdf(Binomial(n, pcon), W[i]) for i = 1:n_visits)
    
    tp = t[end]
    Wp = W[end]
    t = @view(t[1:end-1])
    W = @view(W[1:end-1])
    _, β, z, Γ = solve_logistic(tp, Wp, t, W, n, βu, zu)
    llogistic = normalized_log_likelihood(β, z, Γ, tp, Wp, t, W, n)
    return llogistic - lcon
end