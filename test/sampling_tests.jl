using Random, Distributions
using Plots
using AdaptiveSurveillance

obs, unobs = create_system_states(L, n, 150, [0 typemax(Int64)],
    (t, Γ) -> logistic_prevalance(β_true, logit(p0_true), Γ, t))

astate = AStateIsotonic(α)

# Single Replications


# Probability of Cuessful Detection

# @time iters, la, false_alarm, delay, test_data, locations_visited, ntimes_visisted, 
# last_time_visited, z, w, prevalance_history = replication(L, [0 typemax(Int64)], ones(L) * p0_true, p_sequence, n,
#     astat_isotonic, α, tpolicy_evsi, tstate_evsi(),
#     rng1 = MersenneTwister(1), rng2 = MersenneTwister(2), maxiters = 100);

# # plot(1:iters-1, hcat(z[1:iters], repeat([log(α)], iters-1)), xlabel = "Week", ylabel = "Alarm Statistic", 
# #     label = [permutedims(["Location $l" for l = 1:L])... "Threshold"], legend=:topleft)

# plot(2:(iters-1), hcat(z[3:iters, :], repeat([log(α)], iters-2)), xlabel = "Week", ylabel = "Alarm Statistic", 
#     label = [permutedims(["Location $l" for l = 1:L])... "Threshold"], legend=:topleft)