using StatsFuns
using Distributions
using Isotonic

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

