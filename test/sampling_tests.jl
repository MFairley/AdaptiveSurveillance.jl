using Random, Distributions
using Plots
using StatsFuns
using AdaptiveSurveillance

# Problem Set Up
# System State
obs = StateObservable(L, n, maxiters)
unobs = StateUnobservable(β_true_L, p0_true_L, L, lO, Γ_lO)

# Alarm State
# astate = AStateIsotonic(α)
astate = AStateLogisticTopr(α, βu, p0u, r, L)

# Sampling Policy Tests
# Constant
tstate_constant = TStateConstant(1)

# Random
tstate_random = TStateRandom()

# Thompson Sampling
tstate_thompson = TStateThompson(ones(L, 2))

# Logistic Clairvoyance
tstate_evsi_clairvoyance = TStateEVSIClairvoyant(unobs)

# Logistic Profile
tstate_evsi = TStateEVSI(βu, p0u)
evsi_test = replication(obs, unobs, astate, tstate_evsi)

# Calibration
_ = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_constant, K = K, maxiters = calibration_maxiters)
_ = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_random, K = K, maxiters = calibration_maxiters)
_ = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_thompson, K = K, maxiters = calibration_maxiters)
_ = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_evsi_clairvoyance, K = K, maxiters = calibration_maxiters)
_ = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_evsi, K = K, maxiters = calibration_maxiters)

# Alarm Time Distributions
atd_constant = alarm_time_distribution(K, obs, unobs, astate, tstate_constant, save_path)
atd_random = alarm_time_distribution(K, obs, unobs, astate, tstate_random, save_path)
atd_thompson = alarm_time_distribution(K, obs, unobs, astate, tstate_thompson, save_path)
atd_evsi_clairvoyant = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi_clairvoyance, save_path)
@time atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi, save_path)