using StatsFuns
using Distributions
import DataStructures

abstract type AState end

struct AStateIsotonic <: AState
    α::Float64
    name::String
    AStateIsotonic(α) = new(α, "isotonic")
    AStateIsotonic(α, name) = new(α, name) # needed for @set, internal use only
end

function get_curr_stat(l::Int64, astate::AStateIsotonic)
    #return 0.0
    return -Inf
end

function set_curr_stat(l::Int64, v::Float64, astate::AStateIsotonic)
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
    p0u::Float64
    zu::Float64
    name::String
    AStateLogistic(α, βu, p0u) = new(α, βu, p0u, logit(p0u), "logistic")
    AStateLogistic(α, βu, p0u, zu, name) = new(α, βu, p0u, zu, name) # needed for @set, internal use only
end

function get_curr_stat(l::Int64, astate::AStateLogistic)
    #return 0.0
    return -Inf
end

function set_curr_stat(l::Int64, v::Float64, astate::AStateLogistic)
end

function reset(astate::AStateLogistic)
end

function afunc(l::Int64, obs::StateObservable, astate::AStateLogistic)
    value = astat_logistic(obs.t[l], obs.W[l], obs.n, astate.βu, astate.zu)
    #threshold = log(astate.α)
    #println("$value >? $threshold")
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
    _, β, z, Γ = solve_logistic_mt(tp, Wp, ts, Ws, n, βu, zu)
    llogistic = normalized_log_likelihood(β, z, Γ, tp, Wp, t, W, n)
    return llogistic - lcon
end

struct AStateLogisticTopr <: AState
    α::Float64
    βu::Float64
    p0u::Float64
    zu::Float64
    r::Int64
    L::Int64
    curr_stat::Vector{Float64}
    name::String
    AStateLogisticTopr(α, βu, p0u, r, L) = r > L ? error("r cannot be more than L") : new(α, βu, p0u, logit(p0u), r, L, zeros(L), "logistic_topr")
    AStateLogisticTopr(α, βu, p0u, zu, r, L, curr_stat, name) = new(α, βu, p0u, zu, r, L, curr_stat, name) # needed for @set, internal use only
end

function get_curr_stat(l::Int64, astate::AStateLogisticTopr)
    return astate.curr_stat[l]
end

function set_curr_stat(l::Int64, v::Float64, astate::AStateLogisticTopr)
    astate.curr_stat[l] = v
end

function reset(astate::AStateLogisticTopr)
    #astate.curr_stat .= 0.0
    astate.curr_stat .= -Inf	
end

function afunc(l::Int64, obs::StateObservable, astate::AStateLogisticTopr)
    astate.curr_stat[l] = astat_logistic(obs.t[l], obs.W[l], obs.n, astate.βu, astate.zu)
    #nlargest = DataStructures.nlargest(astate.r, astate.curr_stat)
    #println("nlargest: $nlargest, $(nlargest==[0.0, 0.0])")
    #value = logsumexp(DataStructures.nlargest(astate.r, astate.curr_stat))
    #threshold = log(astate.α)
    #println("$value >? $threshold")
    #if all(DataStructures.nlargest(astate.r, astate.curr_stat) .== 0)
    #    return false
    #end

    return logsumexp(DataStructures.nlargest(astate.r, astate.curr_stat)) > log(astate.α)
end