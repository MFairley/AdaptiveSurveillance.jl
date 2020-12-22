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