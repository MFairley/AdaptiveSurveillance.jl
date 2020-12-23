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

# Random
tstate_random = TStateRandom()

# Thompson Sampling
tstate_thompson = TStateThompson(ones(L, 2))

# Logistic Profile
tstate_evsi = TStateEVSI()

# PSD
K = 2
atd_constant = alarm_time_distribution(K, obs, unobs, astate, tstate_constant)
atd_random = alarm_time_distribution(K, obs, unobs, astate, tstate_random)
atd_thompson = alarm_time_distribution(K, obs, unobs, astate, tstate_thompson)
@time atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi)