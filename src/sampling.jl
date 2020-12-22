### SEARCH POLICIES
struct tstate_const
    l::Int64
end

function tpolicy_constant(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
    return tstate.l
end

function tpolicy_random(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
    return rand(rng, 1:L) # using rng breaks multi-threading here, to do: file github issue for this
end

struct tstate_thompson
    beta_parameters::Array{Float64, 2}
end

function tpolicy_thompson(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
    if t > 1
        l = locations_visited[t - 1] # last location visited
        beta_parameters[l, 1] += test_data[t - 1, l]
        beta_parameters[l, 2] += n - test_data[t - 1, l]
    end
    d = [Beta(tstate.beta_parameters[l, 1], tstate.beta_parameters[l, 2]) for l = 1:L]
    s = rand.(rng, d)
    l = argmax(s)
    return argmax(s)
end

struct tstate_evsi
    # β_max::Float64
    # p0_max::Float64
end

struct SearchV{T, V, F} <: AbstractVector{T}
    v::V
    f::F
    SearchV{T}(v::V, f::F) where {T, V, F} = new{T, V, F}(v, f)
end
Base.size(x::SearchV) = size(x.v)
Base.axes(x::SearchV) = axes(x.v)
Base.@propagate_inbounds Base.getindex(x::SearchV, I...) = x.f(x.v[I...])

function check_astat(i, n, astat, α, test_data, locations_visited, l, t)
    test_data[t, l] = i
    locations_visited[t] = l
    z_candidate, _ = astat(n, @views(test_data[1:t, l][locations_visited[1:t] .== l])) # this is the main bottleneck
    test_data[t, l] = -1
    locations_visited[t] = 0
    return z_candidate > log(α)
end

function tpolicy_evsi(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
    probability_alarm = zeros(L) # estimated log probability that the location will cause an alarm if tested next
    if t > 2 * L # warmup
        for l = 1:L
            f(i) = check_astat(i, n, astat, α, test_data, locations_visited, l, t)
            i = searchsortedfirst(SearchV{Int}(0:n, i -> f(i)), 1) - 1 # note this returns index so need to - 1 to get count
            println("l = $l, i = $i")
            if i > n
                probability_alarm[l] = -Inf
                break
            elseif i == 0
                probability_alarm[l] = Inf
                break
            end
            W = @views(test_data[1:t, l])[locations_visited[1:t] .== l]
            println("l = $l, W = $W")
            times = @views((1:t))[locations_visited[1:t] .== l]
            println("l = $l, times = $times")
            probability_alarm[l] = sum(profile_likelihood(t, times, W, n)[i+1:end])
            println("l = $l, pfa = $(probability_alarm[l])")
        end
        
        return argmax(probability_alarm) # be careful about getting stuck
    end
    return Int(ceil(t / 2))
end