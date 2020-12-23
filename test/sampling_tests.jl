using Random, Distributions
using Plots
using AdaptiveSurveillance

# Problem Set Up
# System State
obs = StateObservable(L, n, 150)
unobs = StateUnobservable([1 typemax(Int64)], (t, Γ) -> logistic_prevalance(β_true, logit(p0_true), Γ, t))

# Alarm State
astate = AStateIsotonic(α)

# Sampling Policy Tests
# Constant
tstate_constant = TStateConstant(1)
res = replication(obs, unobs, astate, tstate_constant)

# Random
tstate_random = TStateRandom()
res = replication(obs, unobs, astate, tstate_random)

# Thompson Sampling
tstate_thompson = TStateThompson(ones(L, 2))
res = replication(obs, unobs, astate, tstate_thompson)

# Logistic Profile
tstate_evsi = TStateEVSI()
# res = replication(obs, unobs, astate, tstate_evsi)

# PSD
K = 2
atd_constant = alarm_time_distribution(K, obs, unobs, astate, tstate_constant)
write_alarm_time_distribution(obs, unobs, atd_constant, joinpath(save_path, "atd_constant.csv"))
atd_random = alarm_time_distribution(K, obs, unobs, astate, tstate_random)
atd_thompson = alarm_time_distribution(K, obs, unobs, astate, tstate_thompson)
@time atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi)