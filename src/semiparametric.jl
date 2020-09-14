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

function replication(L, Γd, Γ::Array{Int64}, p0, p, n, astat::Function, α, tpolicy::Function, tstate; maxiters=1000, warn=true,
    rngd = MersenneTwister(1), rngt = MersenneTwister(2))
    @assert L == length(Γ)
    @assert L == length(Γd)
    @assert L == length(p0)
    @assert L == size(p, 2)
    false_alarm = -1
    test_data = ones(maxiters, L) * -1.0
    locations_visited = zeros(Int64, maxiters)
    ntimes_visisted = zeros(Int64, maxiters, L)
    z = ones(maxiters, L) * -1.0 # alarm statistic
    w = ones(maxiters, L) * -1.0 # posterior probability of states
    for t = 1:maxiters
        l = tpolicy(L, n, α, tstate, rngd, test_data, locations_visited, ntimes_visisted, z, w, t)
        locations_visited[t] = l
        ntimes_visisted[t, l] += 1 + ntimes_visisted[max(1, t - 1), l]
        @views test_data[t, l] = sample_test_data(Γ[l], p0[l], p[:, l], n, rngt, t)

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
        # println(ntimes_visisted[1:t,:])
        # println(locations_visited[1:t])
        # println(test_data[1:t, :])
        if ntimes_visisted[t, l] > 2
            if locations_visited[t] == l
                # println(l)
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
    # println(positive_counts)
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

function tpolicy_constant(L, n, α, tstate, rng, test_data, locations_visited, ntimes_visisted, z, w, t)
    return tstate.l
end

function tpolicy_random(L, n, α, tstate, rng, test_data, locations_visited, ntimes_visisted, z, w, t)
    return rand(1:L) # using rng breaks multi-threading here
end

struct tstate_thompson
    beta_parameters::Array{Float64, 2}
end

function tpolicy_thompson(L, n, α, tstate, rng, test_data, locations_visited, ntimes_visisted, z, w, t)
    if t > 1
        l = locations_visited[t - 1] # last location visisted
        tstate.beta_parameters[l, 1] += test_data[t - 1, l]
        tstate.beta_parameters[l, 2] += n - test_data[t - 1, l]
    end
    d = [Beta(tstate.beta_parameters[l, 1], tstate.beta_parameters[l, 2]) for l = 1:L]
    s = rand.(rng, d)
    l = argmax(s)
    return argmax(s)
end

# function tpolicy_evsi(L, n, α, tstate, rng, test_data, locations_visited, ntimes_visisted, z, w, t)
#     prep_a = zeros(L)
#     if t > 2 * L
#         # println("pD1 = $pD1")
#         # println("pC1 = $pC1")
#         # println("w = $w")
#         # calculate p next time step there is an alarm
#         for l = 1:L
#             ts = collect(1:t)[test_data[1:t, l] .>= 0]
#             for i = 0:n
#                 # println(w[l, :])
#                 z_cand, _ = astat_isotonic(n, t + 1, vcat(ts, t + 1), vcat(test_data[1:t, l][test_data[1:t, l] .>= 0], i))
#                 prep_a[l] += w[l, 1] * pdf(Binomial(n, pD1[l]), i) * (z_cand > log(α))
#                 prep_a[l] += w[l, 2] * pdf(Binomial(n, pC1[l]), i) * (z_cand > log(α))
#             end
#         end
#         # println("prep_a = $prep_a")
#         return sample(1:L, weights(prep_a)), tstate # sampling helps with exploration
#     end
#     return Int(ceil(t / 2)), tstate
# end

### PERFORMANCE METRICS
function alarm_time_distribution(K, L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate; maxiters = 10*52, seed=12)
    alarm_times = zeros(maxiters + 1)
    rngd = MersenneTwister(seed)
    rngt = MersenneTwister(seed)
    Threads.@threads for k = 1:K
        t, _ = replication(L, Γd, Γ, p0, p, n, apolicy::Function, α, tpolicy::Function, tstate, maxiters=maxiters, warn=false,
            rngd = rngd, rngt = rngt)
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
