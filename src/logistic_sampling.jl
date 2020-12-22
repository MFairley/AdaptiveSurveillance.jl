struct tstate_evsi
    # Γd::Array{Geometric{Float64},1} # geometric distribution, cdf is for number of failures before success
    # beta_parameters::Array{Float64, 2}
    # recent_beta_parameters::Array{Float64, 2}
    # a::Int64
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
            if i > n
                probability_alarm[l] = -Inf
                break
            elseif i == 0
                probability_alarm[l] = Inf
                break
            end
            W = @views(test_data[1:t, l])[locations_visited[1:t] .== l]
            
            times = @views((1:t))[locations_visited[1:t] .== l]
            if i > n ÷ 2
                probability_alarm[l] =  future_alarm_log_probability(i, n, t, times, W, n)
            else
                probability_alarm[l] = log1mexp(future_alarm_log_probability(0, i-1, t, times, W, n))
                # log1mexp
            end
        end
        return argmax(probability_alarm) # be careful about getting stuck
    end
    return Int(ceil(t / 2))
end