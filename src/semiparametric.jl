using Statistics
using DelimitedFiles
using Distributions
using RCall

function replication(L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; maxiters=1000)
    test_data = ones(maxiters, L) * -1.0
    false_alarm = true
    for t = 1:maxiters
        la = apolicy(L, Γd, n, α, test_data, t)
        if any(la)
            if any(t .>= Γ[la])
                false_alarm = false
            end
            return t, la, false_alarm, max.(0, t .- Γ)
        end
        l, tstate = tpolicy(test_data, t, tstate)
        @views test_data[t, l] = sample_test_data(Γ[l], p0[l], p[l, :], n, t)
    end
    @warn "The maximum number of time steps, $maxiters, reached."
    return maxiters, zeros(Bool, L), false_alarm, max.(0, maxiters .- Γ)
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
    if t > 2
        for l = 1:L
            @views z = astat_isotonic(n, test_data[1:t, l][test_data[1:t, l] .>= 0])
            thres = logccdf(Γd[l], t - 1) - logcdf(Γd[l], t - 1) + log(α) - log(1 - α)
            # the t - 1 is because the geometric distribution is the number of failures
            if z > thres
                la[l] = true
            end
        end
    end
    return la
end

function astat_isotonic(n, location_test_data)
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
    la = apolicy(L, Γd, n, α, test_data, t)
    la[1:L .!= l] .= false
    return la
end

### SEARCH POLICIES
function tpolicy_constant(test_data, t, tstate)
    return tstate, tstate
end

# to do: thomspon sampling

### PERFORMANCE METRICS
function arl(mode, K, L, Γd, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    run_lengths = zeros(K)
    if mode == 0
        Γ = ones(L) * Inf
    else
        Γ = zeros(L)
    end
    for k = 1:K
        println(k)
        run_lengths[k], _ = replication(L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    end
    return mean(run_lengths), median(run_lengths), var(run_lengths)
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

