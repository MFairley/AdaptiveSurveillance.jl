using Random, Distributions
using Plots
using StatsFuns
using AdaptiveSurveillance

# Problem Set Up
# System State
obs = StateObservable(L, n, maxiters)
unobs = StateUnobservable(β_true_L, p0_true_L, Γ_true_L)

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
tstate_evsi = TStateEVSI(βu, zu)
evsi_test = replication(obs, unobs, astate, tstate_evsi)

# Alarm Time Distributions
const K = 2
atd_constant = alarm_time_distribution(K, obs, unobs, astate, tstate_constant)
atd_random = alarm_time_distribution(K, obs, unobs, astate, tstate_random)
atd_thompson = alarm_time_distribution(K, obs, unobs, astate, tstate_thompson)
@time atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi)