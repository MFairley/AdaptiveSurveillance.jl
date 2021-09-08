using Statistics
using Random
using Distributions
using Base.Threads
using StatsFuns
using DelimitedFiles
using Setfield

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
    β::Vector{Float64} # the transmission rate in each location
    p0::Vector{Float64} # the initial prevalance in each location
    L::Int64 # the number of locations
    lO::Int64 # the location with the outbreak
    Γ_lO::Int64 # the start time of the outbreak
    Γ::Vector{Int64} # the outbreak start time in each location
    StateUnobservable(β, p0, L, lO, Γ_lO) = new(β, p0, L, lO, Γ_lO, [typemax(Int64)*(i != lO)+Γ_lO*(i == lO) for i = 1:L])
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
        
        # Check for an alarm
        !afunc(l, obs, astate) || return t, l, t < unobs.Γ[unobs.lO], max(0, t - unobs.Γ[unobs.lO])
    end
    if warn
        @warn "The maximum number of time steps, $(obs.maxiters), reached for a replication. Exiting"
    end
    return obs.maxiters + 1, 0, -1, -1 # unknown since there was no alarm
end

function sample_test_data(t, l, obs, unobs, rng_system)
    p = logistic_prevalance(unobs.β[l], logit(unobs.p0[l]), unobs.Γ[l], t)
    return rand(rng_system, Binomial(obs.n, p))
end

### PERFORMANCE METRICS
function average_run_length(K::Int64, obs::StateObservable, unobs::StateUnobservable, 
    astate, tstate, tol = 0.5, conf = 0.95)
    times = zeros(K)
    total_count = 0
    for k = 1:K # Threads.@threads 
        t, _, fa, _ = replication(obs, unobs, astate, tstate, k+1, k+2, warn=false, copy=false)
        if fa >= 0
            times[k] = t
            total_count += 1
        end
    end
    if total_count < K
        @warn "The number of timestamps evaluated for ARL is less than K"
    end
    arl = mean(times)
    hw = norminvcdf(1 - (1 - conf) / 2) * std(times) / sqrt(K)
    if hw > tol
        @warn("The half-width for a $(conf) interval of $(hw) is larger than tolerance of $(tol) in the ARL calibratation.")
    end
    return arl, hw
end

function calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate; K = 1000, α1 = 1, α2 = 100, tol = 0.01, maxiters=1000)
    unobs0 = StateUnobservable(unobs.β, unobs.p0, obs.L, 1, typemax(Int64))
    # Bisection search
    i = 0
    α = (α1 + α2) / 2
    astate = @set astate.α = α
    arl, hw = average_run_length(K, obs, unobs0, astate, tstate)

    # Check the target is within the bounds
    astate_low = @set astate.α = α1
    astate_high = @set astate.α = α2
    arl_low, _ = average_run_length(K, obs, unobs0, astate_low, tstate)
    println("ARL, Lower Bound for $(astate.name), $(tstate.name) = $(arl_low)")
    arl_high, _ = average_run_length(K, obs, unobs0, astate_high, tstate)
    println("ARL, Upper Bound for $(astate.name), $(tstate.name) = $(arl_high)")
    @assert arl_low <= target_arl <= arl_high "Target ARL must be within bounds"
    
    while abs(arl - target_arl) > tol
        i += 1
        if arl < target_arl
            α1 = α
        else
            α2 = α
        end
        α = (α1 + α2) / 2
        astate = @set astate.α = α
        arl, hw = average_run_length(K, obs, unobs0, astate, tstate)
        if i >= maxiters
            @warn("Maximum iterations for calibratation reached. Exiting")
            break
        end
    end
    return astate, α, arl, hw
end

function alarm_time_distribution(K::Int64, obs::StateObservable, unobs::StateUnobservable, 
    astate, tstate, filepath::String)
    
    @assert obs.L >= 2
    fn_env = "$(unobs.Γ[1])_$(unobs.p0[1])_$(unobs.p0[2])" # note this is missing many components currently as we assume others constant
    fn_alg = "$(tstate.name)_$(astate.name)"
    filename = joinpath(filepath, "atd_$(fn_alg)_$(fn_env).csv")
    base_line = [unobs.p0[1], unobs.p0[2], unobs.Γ[1], tstate.name, astate.name]

    alarm_times = zeros(Int64, K, 4)
    open(filename, "w") do io
        writedlm(io, ["p1" "p2" "g" "alg" "alarm" "t" "l" "false_alarm" "delay"], ",")
        for k = 1:K # Threads.@threads 
            alarm_times[k, :] .= replication(obs, unobs, astate, tstate, k+1, k+2, warn=false, copy=false)
            writedlm(io, permutedims(vcat(base_line, alarm_times[k, :])), ",")
            flush(io)
        end
    end
    return alarm_times
end