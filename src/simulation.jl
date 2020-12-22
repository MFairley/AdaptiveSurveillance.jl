using Statistics
using Random
using DelimitedFiles
using Distributions
using Base.Threads
using Isotonic
using StatsFuns

function replication(L, Γ::Array{Int64}, p0, p, n, astat::Function, α, tpolicy::Function, tstate; maxiters=1000, warn=true,
    rng1 = MersenneTwister(1), rng2 = MersenneTwister(2))
    @assert L == length(Γ)
    @assert L == length(p0)
    @assert L == size(p, 2)
    tstate = deepcopy(tstate) # this ensures that modifications to state do not affect other runs
    false_alarm = -1
    test_data = ones(Int64, maxiters, L) * -1
    locations_visited = zeros(Int64, maxiters)
    ntimes_visited = zeros(Int64, maxiters, L)
    last_time_visited = ones(Int64, L) * -1
    z = ones(maxiters, L) * -1.0 # alarm statistic
    w = ones(maxiters, L, 2) * -1.0 # posterior log probability of states
    prevalance_history = zeros(maxiters, L) # record of prevalance over time
    for t = 1:maxiters
        l = tpolicy(L, n, astat, α, tstate, rng1, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
        locations_visited[t] = l
        last_time_visited[l] = t
        for j = 1:L
            ntimes_visited[t, j] += ntimes_visited[max(1, t - 1), j] + (j == l)
        end
        test_data[t, l] = sample_test_data(Γ[l], p0[l], @view(p[:, l]), n, rng2, t, l)
        update_prevalance_history!(L, Γ, p0, p, prevalance_history, t) # can comment out if want to save memory

        la = apolicy!(L, n, astat, α, test_data, locations_visited, ntimes_visited, z, w, t)
        if la > 0
            if t >= Γ[la]
                false_alarm = 0
            end
            return t, la, false_alarm, max.(0, t .- Γ), test_data, locations_visited, ntimes_visited, 
                last_time_visited, z, w, prevalance_history
        end
    end
    if warn
        @warn "The maximum number of time steps, $maxiters, reached."
    end
    if all(maxiters + 1 .>= Γ)
        false_alarm = 0
    end
    return maxiters + 1, -1, false_alarm, max.(0, (maxiters + 1) .- Γ), test_data, locations_visited, ntimes_visited, 
        last_time_visited, z, w, prevalance_history
end

function sample_test_data(Γ, p0, p, n, rng, t, l)
    if t < Γ
        return rand(rng, Binomial(n, p0))
    elseif Γ <= t < Γ + length(p)
        return rand(rng, Binomial(n, p[t - Γ + 1]))
    end
    return rand(rng, Binomial(n, p[end]))
end

function update_prevalance_history!(L, Γ, p0, p, prevalance_history, t)
    for l = 1:L
        if t < Γ[l]
            prevalance_history[t, l] = p0[l]
        elseif Γ[l] <= t < Γ[l] + length(p[:, l])
            prevalance_history[t, l] = p[t - Γ[l] + 1, l]
        else
            prevalance_history[t, l] = p[end, l]
        end
    end
end

function apolicy!(L, n, astat::Function, α, test_data, locations_visited, ntimes_visited, z, w, t)
    la = 0
    for l = 1:L
        if ntimes_visited[t, l] > 2
            if locations_visited[t] == l
                z[t, l], w[t, l, :] = astat(n, @views(test_data[1:t, l][locations_visited[1:t] .== l]))
                la = z[t, l] > log(α) ? l : 0
            else
                z[t, l], w[t, l, :] = z[t - 1, l], w[t - 1, l, :]
            end
        else
            z[t, l] = 0.0
            w[t, l, :] = [log(0.5), log(0.5)]
        end
    end
    return la
end



### PERFORMANCE METRICS
function alarm_time_distribution(K, L, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; maxiters = 10*52, seed=12)
    alarm_times = zeros(maxiters + 1)
    rng1 = [MersenneTwister(seed + i) for i = 1:Threads.nthreads()]
    rng2 = [MersenneTwister(seed + 1 + i) for i = 1:Threads.nthreads()]
    Threads.@threads for k = 1:K
        t, _ = replication(L, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate, maxiters=maxiters, warn=false,
            rng1 = rng1[Threads.threadid()], rng2 = rng2[Threads.threadid()])
        alarm_times[t] += 1
    end
    return alarm_times
end

function probability_successfull_detection_l(K, T, d, l, L, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; conf_level=0.95)
    z_score = quantile(Normal(0, 1), 1 - (1 - conf_level)/2)
    post_detections = zeros(T)
    successful_detections = zeros(T)
    for t = 1:T
        Γ = ones(Int64, L) * typemax(Int64)
        Γ[l] = t
        alarm_times = alarm_time_distribution(K, L, Γ, p0, p, n, apolicy, α, tpolicy, tstate; maxiters =  t + d + 1)
        post_detections[t] = sum(alarm_times[t:end])
        successful_detections[t] = sum(alarm_times[t:(t + d)])
    end
    psd = successful_detections ./ post_detections
    hw = z_score .* sqrt.(psd .* (1 .- psd) ./ post_detections)
    return psd, hw
end