using StatsBase

abstract type TState end

struct TStateConstant <: TState
    l::Int64
    name::String
    TStateConstant(l) = new(l, "constant")
end

function reset(tstate::TStateConstant)
end

function tfunc(t, obs, astate, tstate::TStateConstant, rng_test)
    return tstate.l
end

struct TStateRandom <: TState
    name::String
    TStateRandom() = new("random")
end

function reset(tstate::TStateRandom)
end

function tfunc(t, obs, astate, tstate::TStateRandom, rng_test)
    return rand(rng_test, 1:obs.L)
end

struct TStateThompson <: TState
    beta_parameters::Array{Float64, 2}
    name::String
    TStateThompson(beta_parameters) = new(beta_parameters, "thompson")
end

function reset(tstate::TStateThompson)
    tstate.beta_parameters .= 1
end

function tfunc(t, obs, astate, tstate::TStateThompson, rng_test)
    if t > 1
        l = obs.x[t-1] # last location visited
        tstate.beta_parameters[l, 1] += obs.W[l][end]
        tstate.beta_parameters[l, 2] += obs.n - obs.W[l][end]
    end
    lmax = 1
    smax = 0.0
    for l = 1:obs.L
        d = Beta(tstate.beta_parameters[l, 1], tstate.beta_parameters[l, 2])
        s = rand(rng_test, d)
        if s > smax
            smax = s
            lmax = l
        end
    end
    return lmax
end

struct TStateEVSI <: TState
    βu::Float64
    p0u::Float64
    zu::Float64
    name::String
    TStateEVSI(βu, p0u) = new(βu, p0u, logit(p0u), "evsi_$(βu)_$(p0u)")
end

function reset(tstate::TStateEVSI)
end

struct SearchV{T, V, F} <: AbstractVector{T}
    v::V
    f::F
    SearchV{T}(v::V, f::F) where {T, V, F} = new{T, V, F}(v, f)
end
Base.size(x::SearchV) = size(x.v)
Base.axes(x::SearchV) = axes(x.v)
Base.@propagate_inbounds Base.getindex(x::SearchV, I...) = x.f(x.v[I...])

function check_astat(i, l, obs, astate)
    obs.W[l][end] = i
    alarm = afunc(l, obs, astate)
    return alarm
end

function find_threshold(t, l, obs, astate)
    curr_astat = get_curr_stat(l, astate) 
    update!(t, l, 0, obs)
    f(i) = check_astat(i, l, obs, astate)
    i = searchsortedfirst(SearchV{Int}(0:obs.n, i -> f(i)), 1) - 1
    set_curr_stat(l, curr_astat, astate)
    reverse!(l, obs)
    return i
end

function tfunc(t, obs, astate, tstate::TStateEVSI, rng_test)
    # p_alarm_max = 0.0
    # l_alarm_max = 1
    p_alarm = zeros(obs.L)
    if t > 2 * obs.L # warmup
        for l = 1:obs.L
            i = find_threshold(t, l, obs, astate)
            p_alarm_l = 0.0
            if i > obs.n
                p_alarm_l = 0.0
            elseif i == 0
                p_alarm_l = 1.0
            else
                pl = profile_likelihood(t, obs.t[l], obs.W[l], obs.n, tstate.βu, tstate.zu) # allocates memory
                for j = i+1:obs.n
                    p_alarm_l += pl[j]
                end
            end
            p_alarm[l] = p_alarm_l
            # println("t = $t, l = $l, times = $past_times, W = $past_counts, prob = $(probability_alarm[l])")
        end
        return argmax(p_alarm)#sample(rng_test, 1:obs.L, weights(p_alarm))
    end
    return Int(ceil(t / 2))
end

struct TStateEVSIClairvoyant <: TState
    unobs::StateUnobservable
    name::String
    TStateEVSIClairvoyant(unobs) = new(unobs, "evsi_clairvoyant")
end

function reset(tstate::TStateEVSIClairvoyant)
end

function tfunc(t, obs, astate, tstate::TStateEVSIClairvoyant, rng_test)
    p_alarm = zeros(obs.L)
    if t > 2 * obs.L # warmup
        for l = 1:obs.L
            i = find_threshold(t, l, obs, astate)
            p_alarm_l = 0.0
            if i > obs.n
                p_alarm_l = 0.0
            elseif i == 0
                p_alarm_l = 1.0
            else
                p = logistic_prevalance(tstate.unobs.β[l], logit(tstate.unobs.p0[l]), tstate.unobs.Γ[l], t)
                p_alarm_l = sum(pdf(Binomial(obs.n, p), j) for j = i:obs.n)
            end
            p_alarm[l] = p_alarm_l
            # println("t = $t, l = $l, i = $i, prob = $(probability_alarm[l])")
        end
        return argmax(p_alarm)#sample(rng_test, 1:obs.L, weights(p_alarm))
    end
    return Int(ceil(t / 2)) # this is needed otherwise will get stuck
end