using Statistics
using Random
using DelimitedFiles
using Distributions
using Base.Threads
using Isotonic
using StatsFuns

struct StateObservable
    L::Int64 # number of locations
    n::Int64 # number of tests in each time step
    maxiters::Int64 # the maximum number of iters we can do
    x::Array{Int64} # location visited at each time step (our decision)
    W::Array{Int64} # number of positive tests observed
end

function StateObservable(L, n, maxiters)
    StateObservable(L, n, maxiters, ones(Int64, maxiters) * -1, ones(Int64, maxiters) * -1)
end

struct StateUnobservable
    Γ::Array{Int64}
    p::Function # returns prevalance at a given time
end

# input states should be fresh
function replication(obs::StateObservable, unobs::StateUnobservable, astate, tstate;
    seed_system::Int64=1,
    seed_test::Int64=1,
    warn::Bool=true,
    copy::Bool=true)

    # this prevents the replication from modifying arrays in the states
    if copy
        obs = deepcopy(obs)
        unobs = deepcopy(unobs)
        astate = deepcopy(astate)
        tstate = deepcopy(tstate)
    end

    rng_system = MersenneTwister(seed_system)
    rng_test = MersenneTwister(seed_test)

    for t = 1:obs.maxiters
        # Sample from a location and observe positive count
        obs.x[t] = tfunc(t, obs, astate, afunc, tstate, rng_test)
        obs.W[t] = sample_test_data(t, obs.x[t], obs, unobs, rng_system)

        # Check for an alarm in that location (assume alarm static for unchanged locations)
        !afunc(t, obs, astate) || return t, obs.x[t], t < unobs.Γ[obs.x[t]], max(0, t - unobs.Γ[obs.x[t]])
    end
    if warn
        @warn "The maximum number of time steps, $maxiters, reached."
    end
    return maxiters + 1, 0, -1, -1 # unknown since there was no alarm
end

function sample_test_data(t, l, obs, unobs, rng_system)
    p = unobs.p(t, unobs.Γ[l])
    return rand(rng_system, Binomial(obs.n, p))
end

### PERFORMANCE METRICS
function alarm_time_distribution(K::Int64, obs::StateObservable, unobs::StateUnobservable,
    astate, afunc::Function, 
    tstate, tfunc::Function)
    
    alarm_times = zeros(obs.maxiters + 1) 
    Threads.@threads for k = 1:K
        t, _ = replication(obs, unobs, k+1, astate, afunc, k+2, tstate, tfunc, k+3, warn=false)
        alarm_times[t] += 1
    end
    return alarm_times # missing information about which location has the alarm
end

function probability_successfull_detection_l(K::Int64, l::Int64, d::Int64, obs::StateObservable, unobs::StateUnobservable,
    astate, afunc::Function, 
    tstate, tfunc::Function;
    conf_level=0.95)

    z_score = quantile(Normal(0, 1), 1 - (1 - conf_level)/2)
    Γ = unobs.Γ[l]
    @assert(Γ + d <= obs.maxiters)

    alarm_times = alarm_time_distribution(K, obs, unobs, astate, afunc, tstate, tfunc)
    post_detections = sum(alarm_times[Γ:end])
    successful_detections = sum(alarm_times[Γ:(Γ + d)])
    psd = successful_detections / post_detections
    hw = z_score * sqrt(psd * (1 - psd) / post_detections)
    
    return psd, hw
end