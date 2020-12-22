using Random, Distributions
using Plots
using AdaptiveSurveillance

@time iters, la, false_alarm, delay, test_data, locations_visited, ntimes_visisted, 
last_time_visited, z, w, prevalance_history = replication(L, [0 typemax(Int64)], ones(L) * p0_true, p_sequence, n,
    astat_isotonic, α, tpolicy_evsi, tstate_evsi(),
    rng1 = MersenneTwister(1), rng2 = MersenneTwister(2), maxiters = 100);

# plot(1:iters-1, hcat(z[1:iters], repeat([log(α)], iters-1)), xlabel = "Week", ylabel = "Alarm Statistic", 
#     label = [permutedims(["Location $l" for l = 1:L])... "Threshold"], legend=:topleft)

plot(2:(iters-1), hcat(z[3:iters, :], repeat([log(α)], iters-2)), xlabel = "Week", ylabel = "Alarm Statistic", 
    label = [permutedims(["Location $l" for l = 1:L])... "Threshold"], legend=:topleft)

Wt = [1, 2, 4, 4, 3, 2, 2, 2, 0, 4, 3, 2, 2, 2, 4, 2, 1]
tt = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# AdaptiveSurveillance.profile_log_likelihood(0, 1, 20, tt, Wt, n)
AdaptiveSurveillance.profile_likelihood(20, tt, Wt, n)[1:2]