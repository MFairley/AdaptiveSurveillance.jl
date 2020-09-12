using Statistics
using LinearAlgebra
using DelimitedFiles
using Distributions
using RCall

function replication(L, Γd, Γ::Array{Int64}, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; maxiters=1000, warn=true)
    test_data = ones(maxiters, L) * -1.0
    false_alarm = true
    z = ones(maxiters, L) * -1.0
    thres = ones(maxiters, L) * -1.0
    for t = 1:maxiters
        la, z[t, :], thres[t, :] = apolicy(L, Γd, n, α, test_data, t)
        if any(la)
            if all(t .>= Γ[la])
                false_alarm = false
            end
            return t, la, false_alarm, max.(0, t .- Γ), test_data, z, thres
        end
        l, tstate = tpolicy(test_data, t, tstate)
        @views test_data[t, l] = sample_test_data(Γ[l], p0[l], p[:, l], n, t)
    end
    if warn
        @warn "The maximum number of time steps, $maxiters, reached."
    end
    return maxiters, zeros(Bool, L), false_alarm, max.(0, maxiters .- Γ), test_data, z, thres
end

function sample_test_data(Γ, p0, p, n, t)
    if t < Γ
        return rand(Binomial(n, p0))
    elseif Γ <= t < Γ + length(p)
        return rand(Binomial(n, p[t - Γ + 1]))
    end
    return rand(Binomial(n, p[end]))
end

### ALARM POLICIES
function apolicy_isotonic(L, Γd, n, α, test_data, t)
    la = zeros(Bool, L)
    z = zeros(L)
    thres = zeros(L)
    if t > 2
        for l = 1:L
            @views z[l] = astat_isotonic(n, test_data[1:t, l][test_data[1:t, l] .>= 0])
            thres[l] = logccdf(Γd[l], t - 1) - logcdf(Γd[l], t - 1) + log(α) - log(1 - α)
            # the t - 1 is because the geometric distribution is the number of failures
            if z[l] > thres[l]
                la[l] = true
            end
        end
    end
    return la, z, thres
end

function astat_isotonic(n, location_test_data)
    @assert all(0 .<= location_test_data .<= n)
    n_visits = length(location_test_data)
    if n_visits == 0
        return log(1)
    end
    y = 2 * asin.(sqrt.(location_test_data ./ n))
    ir = rcopy(R"isotone::gpava(y = $y)"[:x])
    piso = sin.(ir ./ 2).^2
    pcon = sum(location_test_data) / (n * n_visits)
    liso = sum([logpdf(Binomial(n, piso[i]), location_test_data[i]) for i = 1:n_visits])
    lcon = sum([logpdf(Binomial(n, pcon), location_test_data[i]) for i = 1:n_visits])
    return liso - lcon
end

function apolicy_constant(L, Γd, n, α, test_data, t, apolicy::Function, l)
    la, z, thres = apolicy(L, Γd, n, α, test_data, t)
    la[1:L .!= l] .= false
    return la, z, thres
end

### SEARCH POLICIES
function tpolicy_constant(test_data, t, tstate)
    return tstate, tstate
end

# to do: thomspon sampling, EVSI method

### PERFORMANCE METRICS
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
