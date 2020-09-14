using Statistics
using Random
using DelimitedFiles
using Distributions
using Base.Threads
using Isotonic
using StatsFuns
using DataFrames
using GLM
using StatsBase

function replication(L, Γd, Γ::Array{Int64}, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; maxiters=1000, warn=true,
    rng = MersenneTwister(12))
    test_data = ones(maxiters, L) * -1.0
    false_alarm = true
    z = ones(maxiters, L) * -1.0
    thres = ones(maxiters, L) * -1.0
    l = 1
    for t = 1:maxiters
        la, z[t, :], thres[t, :], w, pD1, pC1 = apolicy(L, Γd, n, α, test_data, t)
        if any(la)
            if all(t .>= Γ[la])
                false_alarm = false
            end
            return t, la, false_alarm, max.(0, t .- Γ), test_data, z, thres
        end
        l, tstate = tpolicy(rng, L, n, l, test_data, t, tstate, w, pD1, pC1, α)
        @views test_data[t, l] = sample_test_data(Γ[l], p0[l], p[:, l], n, rng, t)
    end
    if warn
        @warn "The maximum number of time steps, $maxiters, reached."
    end
    if all(maxiters + 1 .>= Γ)
        false_alarm = false
    end
    return maxiters + 1, ones(Bool, L), false_alarm, max.(0, (maxiters + 1) .- Γ), test_data, z, thres
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
function apolicy_isotonic(L, Γd, n, α, test_data, t)
    la = zeros(Bool, L)
    z = zeros(L)
    thres = zeros(L)
    w = ones(L, 2) * 0.5
    pD1 = zeros(L)
    pC1 = zeros(L)
    if t > 2
        for l = 1:L
            ts = collect(1:t)[test_data[1:t, l] .>= 0]
            z[l], w[l, :], pD1[l], pC1[l] = astat_isotonic(n, t, ts, @view(test_data[1:t, l][test_data[1:t, l] .>= 0]), print=true)
            
            # thres[l] = logccdf(Γd[l], t - 1) - logcdf(Γd[l], t - 1) + log(α) - log(1 - α)
            thres[l] = log(α)
            # the t - 1 is because the geometric distribution is the number of failures
            if z[l] > thres[l]
                la[l] = true
            end
            println(test_data[1:t, l][test_data[1:t, l] .>= 0])
        end
        # println("$t : $pD1")
        
        println("$t : $pC1")
    end
    return la, z, thres, w, pD1, pC1
end

function astat_isotonic(n, t, times, location_test_data; print=false)
    @assert all(0 .<= location_test_data .<= n)
    n_visits = length(location_test_data)
    if n_visits < 2
        return log(1), [0.5, 0.5], 0.0, 0.0
    end
    
    # fit a logistic curve instead - could have some error associated with it
    
    # piso = predict(logit)
    # println(logit)
    # println(data)
    # println(piso)
    

    # estimate piso using logistic regression
    # data = DataFrame(t = time_steps, count = location_test_data)
    # logit = glm(@formula(t ~ count), data, Binomial(), LogitLink())
    y = 2 * asin.(sqrt.(location_test_data ./ n))
    ir = isotonic_regression!(y) # could consider replacing with logistic regression
    piso = sin.(ir ./ 2).^2

    data = DataFrame(t=times, ps=piso)
    logit = glm(@formula(ps ~ t), data, Binomial(), LogitLink())
    if print
        println(data)
    end
    new_data = DataFrame(t=[t])


    pcon = sum(location_test_data) / (n * n_visits)
    liso = sum([logpdf(Binomial(n, piso[i]), location_test_data[i]) for i = 1:n_visits])
    lcon = sum([logpdf(Binomial(n, pcon), location_test_data[i]) for i = 1:n_visits])

    # # return projected piso for next timestep using exponential regression
    # beta = cov(1:n_visits, log.(piso .+ 1e-3)) / var(1:n_visits)
    # alpha = mean(log.(piso .+ 1e-3)) - beta * mean(1:n_visits)
    # piso_tp1 = min(exp(alpha + beta * (n_visits + 1)), 1.0)
    
    # println(piso_tp1) # issue when estimate is 0.0, add some eps, also consider fitting a logit model
    # println(piso)
    # println(location_test_data)
    # println(Float64(predict(logit, new_data)[1]))
    return liso - lcon, softmax([lcon, liso]), pcon, predict(logit, new_data)[1]
end

function apolicy_constant(L, Γd, n, α, test_data, t, apolicy::Function, l)
    la, z, thres = apolicy(L, Γd, n, α, test_data, t)
    la[1:L .!= l] .= false
    return la, z, thres
end

### SEARCH POLICIES
function tpolicy_constant(rng, L, n, l, test_data, t, tstate, w, pD1, pC1, α)
    return tstate, tstate
end

function tpolicy_random(rng, L, n, l, test_data, t, tstate, w, pD1, pC1, α)
    return rand(1:L), tstate # using rng here is apparently not threadsafe
end

function tpolicy_thompson(rng, L, n, l, test_data, t, tstate, w, pD1, pC1, α)
    if t > 1
        @assert test_data[t-1, l] >= 0
        tstate[l, 1] += test_data[t-1, l]
        tstate[l, 2] += n - test_data[t-1, l]
    end
    d = [Beta(tstate[l, 1], tstate[l, 2]) for l = 1:size(tstate, 1)]
    s = rand.(d)
    return argmax(s), tstate
end

function tpolicy_evsi(rng, L, n, l, test_data, t, tstate, w, pD1, pC1, α)
    prep_a = zeros(L)
    if t > 2 * L
        # println("pD1 = $pD1")
        # println("pC1 = $pC1")
        # println("w = $w")
        # calculate p next time step there is an alarm
        for l = 1:L
            ts = collect(1:t)[test_data[1:t, l] .>= 0]
            for i = 0:n
                # println(w[l, :])
                z_cand, _ = astat_isotonic(n, t + 1, vcat(ts, t + 1), vcat(test_data[1:t, l][test_data[1:t, l] .>= 0], i))
                prep_a[l] += w[l, 1] * pdf(Binomial(n, pD1[l]), i) * (z_cand > log(α))
                prep_a[l] += w[l, 2] * pdf(Binomial(n, pC1[l]), i) * (z_cand > log(α))
            end
        end
        # println("prep_a = $prep_a")
        return sample(1:L, weights(prep_a)), tstate # sampling helps with exploration
    end
    return Int(ceil(t / 2)), tstate
end

# to do: thomspon sampling, EVSI method

### PERFORMANCE METRICS
function alarm_time_distribution(K, L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; maxiters = 10*52, seed=12)
    alarm_times = zeros(maxiters + 1)
    rng = MersenneTwister(seed)
    Threads.@threads for k = 1:K
        t, _ = replication(L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate, maxiters=maxiters, warn=false, rng = rng)
        alarm_times[t] += 1
    end
    return alarm_times
end

function probability_successfull_detection_l(K, T, d, l, L, Γd, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    post_detections = zeros(T)
    successful_detections = zeros(T)
    for t = 1:T
        Γ = ones(Int64, L) * typemax(Int64)
        Γ[l] = t
        alarm_times = alarm_time_distribution(K, L, Γd, Γ, p0, p, n, apolicy, α, tpolicy, tstate; maxiters =  t + d + 1)
        post_detections[t] = sum(alarm_times[t:end])
        successful_detections[t] = sum(alarm_times[t:(t + d)])
    end
    return successful_detections ./ post_detections
end

### old below
function arl(mode, K, L, Γd, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    run_lengths = zeros(K)
    if mode == 0
        Γ = ones(Int64, L) * typemax(Int64)
    else
        Γ = zeros(Int64, L)
    end

    for k = 1:K
        run_lengths[k], _ = replication(L, Γd, Γ, p0, p, n, apolicy, α, tpolicy, tstate)
    end
    return mean(run_lengths), median(run_lengths), var(run_lengths)
end

function predictive_value(eps, T, L, Γd, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate, miniters=1000)
    @assert T > 2
    alarm_times = ones(T - 2, 2) # uninformative prior
    i = 1
    converged = false
    while i <= miniters || !converged
        Γ = rand.(Γd) # could do importance sampling on this but problem is that it is high dimensional
        t, _, fa, _ = replication(L, Γd, Γ, p0, p, n, apolicy, α, tpolicy, tstate, maxiters = T, warn=false)
        alarm_times[t - 2, Int(fa) + 1] += 1
        if all(varbeta(alarm_times) .<= eps)
            converged = true
        else
            converged = false
        end
        i += 1
        if i % 10000 == 0
            println(alarm_times)
        end
    end
    return alarm_times
end

function predictive_value_ratio(eps, T, L, Γd, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; miniters=1000)
    count = 0
    alarm_times = zeros(T)
    alarm_times_location = zeros(T)
    i = 1
    # converged = false
    while i <= miniters # || !converged
        count += 1
        i += 1
        Γ = rand.(Γd)
        t, _, fa, _ = replication(L, Γd, Γ, p0, p, n, apolicy, α, tpolicy, tstate, maxiters = T + 1, warn=false)
        if t <= T
            alarm_times[t - 2] += 1
            if !fa
                alarm_times_location[t - 2] += 1
            end
        end
    end
    return alarm_times, alarm_times_location, (alarm_times_location ./ count) ./ (alarm_times ./ count)
end

function alarm_density(K, T, L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    alarm_counts = zeros(T + 1)
    for k = 1:K
        t, _ = replication(L, Γd, Γ, p0, p, n, apolicy, α, tpolicy, tstate, maxiters = T + 1, warn=false)
        alarm_counts[t] += 1
    end
    return cumsum(alarm_counts[1:T]) ./ K
end

function probability_successful_detection(K, d, l, L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    successful_detections = 0
    post_change_alarms = 0
    Γv = ones(Int64, L) * typemax(Int64)
    Γv[l] = Γ
    for k = 1:K
        t, _, _, delays, _ = replication(L, Γd, Γv, p0, p, n, apolicy, α, tpolicy, tstate, maxiters = Γ + d + 1, warn=false)
        if t >= Γ
            post_change_alarms += 1
            if delays[l] <= d
                successful_detections += 1
            end
        end
    end
    return successful_detections, post_change_alarms
end

function fixedΓ_alarm_distribution(K, d, l, L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    time_counts = zeros(Int64, Γ + d + 1)
    Γv = ones(Int64, L) * typemax(Int64)
    Γv[l] = Γ
    for k = 1:K
        alarm_time, _ = replication(L, Γd, Γv, p0, p, n, apolicy, α, tpolicy, tstate, maxiters=Γ + d + 1, warn=false)
        time_counts[alarm_time] += 1
    end
    return time_counts ./ K
end

# to do: plots

# output writing
function write_results(L, outbreak_starts, alarm_times, location_alarms, false_alarm, delays, file)
    os_header = ["l$(i)_outbreakt".format(i) for i = 1:L]
    la_header = ["l$(i)_alarmt".format(i) for i = 1:L]
    delay_header = ["l$(i)_delay".format(i) for i = 1:L]
    open(file, "w") do io
        writedlm(io, ["alarm_time", la_header..., "false_alarm", delay_header...])
        for i = 1:size(alarm_times, 1)
            writedlm(io, [alarm_times[i], outbreak_starts[i, :], location_alarms[i, :]..., false_alarm[i], delays[i, :]...])
        end
    end
end

function varbeta(p)
    s = sum(p, dims=2)
    return (p[:, 1] .* p[:, 2]) ./ (abs2.(s) .* (s .+ 1))
end
