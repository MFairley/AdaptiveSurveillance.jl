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
    x::Array{Int64} # location visited at each time step (our decision)
    W::Array{Int64} # number of positive tests observed
end

function StateObservable(L, n, maxiters)
    StateObservable(L, n, maxiters, ones(Int64, maxiters) * -1, ones(Int64, maxiters) * -1)
end

function reset(state::StateObservable)
    state.x .= -1
    state.W .= -1
end

struct StateUnobservable
    Γ::Array{Int64} # the outbreak start time in each location
    p::Function # returns prevalance at a given time
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
        obs.x[t] = tfunc(t, obs, astate, afunc, tstate, rng_test)
        obs.W[t] = sample_test_data(t, obs.x[t], obs, unobs, rng_system)

        # Check for an alarm in that location (assume alarm static for unchanged locations)
        !afunc(t, obs, astate) || return t, obs.x[t], t < unobs.Γ[obs.x[t]], max(0, t - unobs.Γ[obs.x[t]])
    end
    if warn
        @warn "The maximum number of time steps, $maxiters, reached."
    end
    return obs.maxiters + 1, 0, -1, -1 # unknown since there was no alarm
end

function sample_test_data(t, l, obs, unobs, rng_system)
    p = unobs.p(t, unobs.Γ[l])
    return rand(rng_system, Binomial(obs.n, p))
end

### PERFORMANCE METRICS
function alarm_time_distribution(K::Int64, obs::StateObservable, unobs::StateUnobservable, 
    astate, tstate)
    
    alarm_times = zeros(Int64, obs.maxiters + 1, obs.L + 1)
    for k = 1:K # Threads.@threads 
        t, l, _ = replication(obs, unobs, astate, tstate, k+1, k+2, warn=false, copy=false)
        alarm_times[t, l + 1] += 1
    end
    return alarm_times
end

function write_alarm_time_distribution(obs, unobs, alarm_times, filename)
    p_sequence = zeros(size(alarm_times, 1), obs.L)
    for l = 1:obs.L
        for t = 1:size(alarm_times, 1)
            p_sequence[t, l] = unobs.p(t, unobs.Γ[l])
        end
    end
    writedlm(filename, hcat(p_sequence, alarm_times), ",")
end

# function probability_successfull_detection_l(K::Int64, l::Int64, d::Int64, obs::StateObservable, unobs::StateUnobservable,
#     astate, tstate;
#     conf_level=0.95)

#     z_score = quantile(Normal(0, 1), 1 - (1 - conf_level)/2)
#     Γ = unobs.Γ[l]
#     @assert(Γ + d <= obs.maxiters)

#     alarm_times = alarm_time_distribution(K, obs, unobs, astate, tstate)
#     post_detections = sum(alarm_times[Γ:end])
#     successful_detections = sum(alarm_times[Γ:(Γ + d)])
#     psd = successful_detections / post_detections
#     hw = z_score * sqrt(psd * (1 - psd) / post_detections)
    
#     return psd, hw
# end