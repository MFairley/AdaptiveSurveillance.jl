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
using Turing
# Turing.setadbackend(:reverse_diff)

function replication(L, Γ::Array{Int64}, p0, p, n, astat::Function, α, tpolicy::Function, tstate; maxiters=1000, warn=true,
    rng1 = MersenneTwister(1), rng2 = MersenneTwister(2))
    @assert L == length(Γ)
    @assert L == length(p0)
    @assert L == size(p, 2)
    tstate = deepcopy(tstate) # this ensures that modifications to state do not affect other runs
    false_alarm = -1
    test_data = ones(maxiters, L) * -1.0
    locations_visited = zeros(Int64, maxiters)
    ntimes_visisted = zeros(Int64, maxiters, L)
    last_time_visited = ones(L) * -1.0
    z = ones(maxiters, L) * -1.0 # alarm statistic
    w = ones(maxiters, L) * -1.0 # posterior probability of states
    for t = 1:maxiters
        l = tpolicy(L, n, astat, α, tstate, rng1, test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)
        locations_visited[t] = l
        last_time_visited[l] = t
        for j = 1:L
            ntimes_visisted[t, j] += ntimes_visisted[max(1, t - 1), j] + (j == l)
        end
        @views test_data[t, l] = sample_test_data(Γ[l], p0[l], p[:, l], n, rng2, t)

        la = apolicy!(L, n, astat, α, test_data, locations_visited, ntimes_visisted, z, w, t)
        if la > 0
            if t >= Γ[la]
                false_alarm = 0
            end
            return t, la, false_alarm, max.(0, t .- Γ), test_data, locations_visited, ntimes_visisted, z, w
        end
    end
    if warn
        @warn "The maximum number of time steps, $maxiters, reached."
    end
    if all(maxiters + 1 .>= Γ)
        false_alarm = 0
    end
    return maxiters + 1, -1, false_alarm, max.(0, (maxiters + 1) .- Γ), test_data, locations_visited, ntimes_visisted, z, w
end

function sample_test_data(Γ, p0, p, n, rng, t)
    if t < Γ
        return rand(rng, Binomial(n, p0))
    elseif Γ <= t < Γ + length(p)
        return rand(rng, Binomial(n, p[t - Γ + 1]))
    end
    return rand(rng, Binomial(n, p[end]))
end

function apolicy!(L, n, astat::Function, α, test_data, locations_visited, ntimes_visisted, z, w, t)
    la = 0
    for l = 1:L
        if ntimes_visisted[t, l] > 2
            if locations_visited[t] == l
                z[t, l], w[t, l] = astat(n, @view(test_data[1:t, l][locations_visited[1:t] .== l]))
                la = z[t, l] > log(α) ? l : 0
            else
                z[t, l], w[t, l] = z[t - 1, l], w[t - 1, l]
            end
        else
            z[t, l] = 0.0
            w[t, l] = 0.5
        end
    end
    return la
end

### ALARM STATISTICS
function astat_isotonic(n, positive_counts)
    @assert all(0 .<= positive_counts .<= n)
    n_visits = length(positive_counts)
    y = 2 * asin.(sqrt.(positive_counts ./ n))
    ir = isotonic_regression!(y)
    piso = sin.(ir ./ 2).^2
    pcon = sum(positive_counts) / (n * n_visits)
    liso = sum([logpdf(Binomial(n, piso[i]), positive_counts[i]) for i = 1:n_visits])
    lcon = sum([logpdf(Binomial(n, pcon), positive_counts[i]) for i = 1:n_visits])
    return liso - lcon, softmax([lcon, liso])[1]
end

### SEARCH POLICIES
struct tstate_const
    l::Int64
end

function tpolicy_constant(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)
    return tstate.l
end

function tpolicy_random(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)
    return rand(rng, 1:L) # using rng breaks multi-threading here, to do: file github issue for this
end

struct tstate_thompson
    beta_parameters::Array{Float64, 2}
end

function beta_update(n, tstate, test_data, l, t)
    tstate.beta_parameters[l, 1] += test_data[t - 1, l]
    tstate.beta_parameters[l, 2] += n - test_data[t - 1, l]
end

function tpolicy_thompson(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)
    if t > 1
        l = locations_visited[t - 1] # last location visisted
        beta_update(n, tstate, test_data, l, t)
    end
    d = [Beta(tstate.beta_parameters[l, 1], tstate.beta_parameters[l, 2]) for l = 1:L]
    s = rand.(rng, d)
    l = argmax(s)
    return argmax(s)
end

struct tstate_evsi
    Γd # geometric distribution, cdf is for number of failures before success
    beta_parameters::Array{Float64, 2}
    fraction_forget::Float64
end

function tpolicy_evsi(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)
    if t > 1
        l = locations_visited[t - 1] # last location visisted
        beta_update(n, tstate, test_data, l, t)
    end
    probability_alarm = zeros(L) # estimated probability that the location will cause an alarm if tested next
    if t > 2 * L # warmup
        for l = 1:L
            t_forget = Int(ceil(tstate.fraction_forget *  (t - 1)))
            tprime = last_time_visited[l] - 1
            
            pDn = w[t - 1, l] * ccdf(tstate.Γd[l], t - 1)
            pDd = (1 - w[t - 1, l]) * cdf(tstate.Γd[l], tprime) + w[t - 1, l] * (cdf(tstate.Γd[l], t - 1) - cdf(tstate.Γd[l], tprime))
            pD = pDn / (pDn + pDd)
            dD = BetaBinomial(n, tstate.beta_parameters[l, 1], tstate.beta_parameters[l, 2])
            
            @views positive_forget = sum(test_data[t_forget:t, l][locations_visited[t_forget:t] .== l])
            @views tests_forget = sum(locations_visited[t_forget:t] .== l) * n
            dC = BetaBinomial(n, 1 + positive_forget, 1 + tests_forget - positive_forget)
            
            for i = 0:n # to do: find first n that causes switch
                test_data[t, l] = i
                locations_visited[t] = l
                z, _ = astat(n, @view(test_data[1:t, l][locations_visited[1:t] .== l]))
                # z = rand(Uniform(log(α) - 10, log(α) + 10))
                if z > log(α)
                    probability_alarm[l] += pD * (isnan(pdf(dD, i)) ? 0.0 : pdf(dD, i))
                    probability_alarm[l] += (1 - pD) * (isnan(pdf(dC, i)) ? 0.0 : pdf(dC, i))
                end
            end
            test_data[t, l] = -1.0
            locations_visited[t] = 0
        end
        # println(probability_alarm)
        @assert all(0 .<= probability_alarm .<= 1.0) # probability need to do log probabilities
        return argmax(probability_alarm) 
    end
    return Int(ceil(t / 2))
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

function probability_successfull_detection_l(K, T, d, l, L, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate)
    post_detections = zeros(T)
    successful_detections = zeros(T)
    for t = 1:T
        Γ = ones(Int64, L) * typemax(Int64)
        Γ[l] = t
        alarm_times = alarm_time_distribution(K, L, Γ, p0, p, n, apolicy, α, tpolicy, tstate; maxiters =  t + d + 1)
        post_detections[t] = sum(alarm_times[t:end])
        successful_detections[t] = sum(alarm_times[t:(t + d)])
    end
    return successful_detections ./ post_detections
end
