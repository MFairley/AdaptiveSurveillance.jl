using Random, Distributions
using Plots
using AdaptiveSurveillance

# Problem Set Up
# System State
obs = StateObservable(L, n, 150)
unobs = StateUnobservable([0 typemax(Int64)], (t, Γ) -> logistic_prevalance(β_true, logit(p0_true), Γ, t))

# Alarm State
astate = AStateIsotonic(α)

# Sampling Policy
# Constant
tstate_constant = TStateConstant(1)
res = replication(obs, unobs, astate, tstate_constant, copy=true)

# Random
tstate_random = TStateRandom()
res = replication(obs, unobs, astate, tstate_random, copy=true)

# Thompson Sampling
tstate_thompson = TStateThompson(ones(L, 2))
res = replication(obs, unobs, astate, tstate_thompson, copy=false)

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