using Distributions
using Plots
using AdaptiveSurveillance

# Setup
const output_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
const K = 1000 # number of reps
const d = 4 # delay for successful detection
const L = 3
const ν = 1 / 52 # approx once per year there is an outbreak
const Γd = [Geometric(ν) for l = 1:L]
const p0 = 0.01 * ones(L)
const β = 4e-6 * 536 * 7
const p = repeat(prevalance_sequence(p0[1], β), 1, L)
const n = 100
α = 0.9 # go up to 1.0, the higher, the less false positives
apolicy_constant_p(L, Γd, n, α, test_data, t) = apolicy_constant(L, Γd, n, α, test_data, t, apolicy_isotonic, 1)

# Debug
# Γ = ones(Int64, L) * typemax(Int64)
# Γ = zeros(Int64, L)
# t, la, false_alarm, delay, test_data, z, thres = replication(L, Γd, Γ, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)
# plot(2:(t-1), hcat(z[3:t], thres[3:t]), label = ["z" "thres"])
# plot(1:(t-1), test_data[1:(t-1), 1])

# Average Run Length 0
# arl0_constant = arl(0, K, L, Γd, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)
# a_grid = 0.05:0.05:0.95
# arl1_alpha = zeros(length(a_grid))
# for (i, α) in enumerate(a_grid)
    # arl1_alpha[i] = arl(1, K, L, Γd, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)[1]
# end

# Alarm Time Distribution
# adist = fixedΓ_alarm_distribution(K, d, 1, L, Γd, 4, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)
alarm_counts = predictive_value(1e-3, 52, L, Γd, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)
