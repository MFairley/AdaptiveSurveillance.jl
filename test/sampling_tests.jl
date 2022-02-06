using Random, Distributions
using Plots
using StatsFuns
using AdaptiveSurveillance

# Problem Set Up
# System State
obs = StateObservable(L, n, maxiters)
unobs = StateUnobservable(β_true_L, p0_true_L, L, lO, Γ_lO)

# Alarm State
astate = AStateIsotonic(α)
# astate = AStateLogistic(α, βu, p0u)
# astate = AStateLogisticTopr(α, βu, p0u, r, L)

# Sampling Policy Tests
# Constant
tstate_constant = TStateConstant(lO)

# Random
tstate_random = TStateRandom()

# Thompson Sampling
tstate_thompson = TStateThompson(ones(L, 2))

# Logistic Clairvoyance
tstate_evsi_clairvoyance = TStateEVSIClairvoyant(unobs)

# Logistic Profile
tstate_evsi = TStateEVSI(βu, p0u)
evsi_test = replication(obs, unobs, astate, tstate_evsi)

# Complete calibration
astatec, αc, arlc, hwc, obs_calibrated, unobs_calibrated = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_constant, maxiters = maxiters_calibrationc, arl_maxiters = Kc)
atd_constant = alarm_time_distribution(Kc, obs_calibrated, unobs_calibrated, astatec, tstate_constant, save_path)
@test isapprox(target_arl, mean(atd_constant[:, 1]), rtol=0.15)

astatec, αc, arlc, hwc, obs_calibrated, unobs_calibrated = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_random, maxiters = maxiters_calibrationc, arl_maxiters = Kc)
atd_random = alarm_time_distribution(Kc, obs_calibrated, unobs_calibrated, astatec, tstate_random, save_path)
@test isapprox(target_arl, mean(atd_random[:, 1]), rtol=0.15)

astatec, αc, arlc, hwc, obs_calibrated, unobs_calibrated = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_thompson, maxiters = maxiters_calibrationc, arl_maxiters = Kc)
atd_thompson = alarm_time_distribution(Kc, obs_calibrated, unobs_calibrated, astatec, tstate_thompson, save_path)
@test isapprox(target_arl, mean(atd_thompson[:, 1]), rtol=0.15)

# Calibration
_ = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_evsi_clairvoyance, maxiters = maxiters_calibration, arl_maxiters = K)
# _ = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate_evsi, maxiters = maxiters_calibration, arl_maxiters = K) # <- runs very slowly

# # Alarm Time Distributions
atd_evsi_clairvoyant = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi_clairvoyance, save_path)
atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi, save_path)

