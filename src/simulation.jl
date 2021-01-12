using Statistics
using Random
using Distributions
using Base.Threads
using StatsFuns
using DelimitedFiles

struct StateObservable
    L::Int64 # number of locations
    n::Int64 # number of tests in each time step
    maxiters::Int64 # the maximum number of iters we can do
    x::Vector{Int64} # which location we visited (our decision)
    t::Vector{Vector{Int64}} # times corresponding to when tests collected
    W::Vector{Vector{Int64}} # number of positive tests observed
    StateObservable(L, n, maxiters) = new(L, n, maxiters, [], [[] for i = 1:L], [[] for i = 1:L])
end

function update!(t, l, W, state::StateObservable)
    append!(state.x, l)
    append!(state.t[l], t)
    append!(state.W[l], W)
end

function reverse!(l, state::StateObservable)
    pop!(state.x)
    pop!(state.t[l])
    pop!(state.W[l])
end

function reset(state::StateObservable)
    empty!(state.x)
    empty!.(state.t)
    empty!.(state.W)
end

struct StateUnobservable
    β::Vector{Float64} # the transmissionr ate in each location
    p0::Vector{Float64} # the initial prevalance in each location
    Γ::Vector{Int64} # the outbreak start time in each location
end

function reset(state::StateUnobservable)
end

function replication(obs::StateObservable, unobs::StateUnobservable, astate, tstate, # declare types here 
    seed_system::Int64=1,
    seed_test::Int64=1;
    warn::Bool=true,
    copy::Bool=false)

    # this may be helpful for threading
    if copy
        obs = deepcopy(obs)
        unobs = deepcopy(unobs)
        astate = deepcopy(astate)
        tstate = deepcopy(tstate)
    end

    reset(obs)
    reset(unobs)
    reset(astate)
    reset(tstate)

    rng_system = MersenneTwister(seed_system)
    rng_test = MersenneTwister(seed_test)

    for t = 1:obs.maxiters
        # Sample from a location and observe positive count
        l = tfunc(t, obs, astate, tstate, rng_test)
        W = sample_test_data(t, l, obs, unobs, rng_system)
        update!(t, l, W, obs)
        
        # Check for an alarm in that location (assume alarm static for unchanged locations)
        !afunc(l, obs, astate) || return t, l, t < unobs.Γ[l], max(0, t - unobs.Γ[l])
    end
    if warn
        @warn "The maximum number of time steps, $(obs.maxiters), reached."
    end
    return obs.maxiters + 1, 0, -1, -1 # unknown since there was no alarm
end

function sample_test_data(t, l, obs, unobs, rng_system)
    p = logistic_prevalance(unobs.β[l], logit(unobs.p0[l]), unobs.Γ[l], t)
    return rand(rng_system, Binomial(obs.n, p))
end

### PERFORMANCE METRICS
function alarm_time_distribution(K::Int64, obs::StateObservable, unobs::StateUnobservable, 
    astate, tstate, filename::String)
    alarm_times = zeros(Int64, K, 4)
    open(filename, "w") do io
        writedlm(io, ["t" "l" "false_alarm" "delay"], ",")
        for k = 1:K # Threads.@threads 
            alarm_times[k, :] .= replication(obs, unobs, astate, tstate, k+1, k+2, warn=false, copy=false)
            writedlm(io, permutedims(alarm_times[k, :]), ",")
        end
    end
    return alarm_times
end