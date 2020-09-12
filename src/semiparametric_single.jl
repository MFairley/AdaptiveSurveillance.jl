module sp_single 

using Statistics
using Random
using DelimitedFiles
using Distributions
using RCall
using Base.Threads
using Isotonic

function replication(Γd, Γ, p0, p, n, apolicy::Function, α; maxiters=10*52, rng = MersenneTwister(12), warn=true)
    test_data = ones(maxiters - 1) * -1.0
    false_alarm = true
    z = ones(maxiters) * -1.0
    thres = ones(maxiters) * -1.0
    for t = 1:maxiters
        la, z[t], thres[t] = apolicy(Γd, n, α, test_data, t)
        if la
            if t >= Γ
                false_alarm = false
            end
            return t, false_alarm, max(0, t - Γ), test_data, z, thres
        end
        if t < maxiters # maxiters is final decision step, no more data collection afterwards
            @views test_data[t] = sample_test_data(Γ, p0, p, n, rng, t)
        end
    end
    if warn
        @warn "The maximum number of time steps, $maxiters, reached."
    end
    if maxiters + 1 >= Γ
        false_alarm = false
    end
    return maxiters + 1, false_alarm, max(0, maxiters + 1 - Γ), test_data, z, thres
end

function sample_test_data(Γ, p0, p, n, rng, t)
    if t < Γ
        return rand(rng, Binomial(n, p0))
    elseif Γ <= t < Γ + length(p)
        return rand(rng, Binomial(n, p[t - Γ + 1]))
    end
    return rand(rng, Binomial(n, p[end]))
end

### ALARM POLICIES
function apolicy_isotonic(Γd, n, α, test_data, t)
    alarm = false
    z = 0.0
    thres = 0.0
    if t > 2
        @views z = astat_isotonic(n, test_data[1:(t-1)])
        # thres = logccdf(Γd, t - 1) - logcdf(Γd, t - 1) + log(α)
        thres = log(α)
        if z > thres
            alarm = true
        end
    end
    return alarm, z, thres
end

function astat_isotonic(n, location_test_data)
    @assert all(0 .<= location_test_data .<= n)
    n_visits = length(location_test_data)
    if n_visits == 0
        return log(1)
    end
    y = 2 * asin.(sqrt.(location_test_data ./ n))
    ir = isotonic_regression!(y)
    piso = sin.(ir ./ 2).^2
    pcon = sum(location_test_data) / (n * n_visits)
    liso = sum([logpdf(Binomial(n, piso[i]), location_test_data[i]) for i = 1:n_visits])
    lcon = sum([logpdf(Binomial(n, pcon), location_test_data[i]) for i = 1:n_visits])
    return liso - lcon
end

### PERFORMANCE METRICS
function alarm_time_distribution(K, Γd, Γ, p0, p, n, apolicy::Function, α; maxiters = 10*52, seed=12)
    alarm_times = zeros(maxiters + 1)
    rng = MersenneTwister(seed)
    Threads.@threads for k = 1:K
        t, _ = replication(Γd, Γ, p0, p, n, apolicy, α, maxiters = maxiters, rng = rng, warn = false)
        alarm_times[t] += 1
    end
    return alarm_times
end

function predictive_value(K, T1, T2, Γd, p0, p, n, apolicy::Function, α)
    predictive_value = zeros(T2)
    for t = T1:T2
        pv_sum = 0.0
        for i = 1:t
            alarm_times = alarm_time_distribution(K, Γd, i, p0, p, n, apolicy, α, maxiters = t + 1)
            pv_sum += alarm_times[t] / K * pdf(Γd, i - 1)
        end
        alarm_times_pfa = alarm_time_distribution(K, Γd, t + 1, p0, p, n, apolicy, α, maxiters = t + 1)
        pfa = (alarm_times_pfa[t] / K) * ccdf(Γd, t - 1)
        predictive_value[t] = pv_sum / (pv_sum + pfa)
    end
    return predictive_value
end

function probability_successfull_detection(K, T, d, Γd, p0, p, n, apolicy::Function, α)
    post_detections = zeros(T)
    successful_detections = zeros(T)
    for t = 1:T
        alarm_times = alarm_time_distribution(K, Γd, t, p0, p, n, apolicy, α, maxiters = t + d + 1)
        post_detections[t] = sum(alarm_times[t:end])
        successful_detections[t] = sum(alarm_times[t:(t + d)])
    end
    return successful_detections ./ post_detections
end

end