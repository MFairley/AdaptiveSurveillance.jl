### SEARCH POLICIES
struct TStateConstant
    l::Int64
end

function tfunc(t, obs, astate, afunc, tstate::TStateConstant, rng_test)
    return tstate.l
end

struct TStateRandom
end

function tfunc(t, obs, astate, afunc, tstate::TStateRandom, rng_test)
    return rand(rng_test, 1:obs.L)
end

struct TStateThompson
    beta_parameters::Array{Float64, 2}
end

function tfunc(t, obs, astate, afunc, tstate::TStateThompson, rng_test)
    if t > 1
        l = obs.x[t - 1] # last location visited
        tstate.beta_parameters[l, 1] += obs.W[t - 1]
        tstate.beta_parameters[l, 2] += obs.n - obs.W[t - 1]
    end
    d = [Beta(tstate.beta_parameters[l, 1], tstate.beta_parameters[l, 2]) for l = 1:obs.L]
    s = rand.(rng_test, d)
    l = argmax(s)
    return argmax(s)
end

struct TStateEVSI
end

struct SearchV{T, V, F} <: AbstractVector{T}
    v::V
    f::F
    SearchV{T}(v::V, f::F) where {T, V, F} = new{T, V, F}(v, f)
end
Base.size(x::SearchV) = size(x.v)
Base.axes(x::SearchV) = axes(x.v)
Base.@propagate_inbounds Base.getindex(x::SearchV, I...) = x.f(x.v[I...])

function check_astat(i, t, l, obs, astate, afunc)
    obs.x[t] = l
    obs.W[t] = i
    alarm = afunc(t, obs, astate)
    obs.x[t] = -1
    obs.W[t] = -1
    return return alarm
end

function tfunc(t, obs, astate, afunc, tstate::TStateEVSI, rng_test)
    probability_alarm = zeros(L)
    if t > 2 * L # warmup
        for l = 1:L
            f(i) = check_astat(i, t, l, obs, astate, afunc)
            i = searchsortedfirst(SearchV{Int}(0:n, i -> f(i)), 1) - 1
            if i > n
                probability_alarm[l] = 0.0
                break
            elseif i == 0
                probability_alarm[l] = 1.0
                break
            end
            past_times = (1:t-1)[obs.x .== l]
            past_counts = obs.W[obs.x .== 1]
            probability_alarm[l] = sum(profile_likelihood(t, past_times, past_counts, obs.n)[i+1:end])
        end
        return argmax(probability_alarm) # be careful about getting stuck
    end
    return Int(ceil(t / 2))
end