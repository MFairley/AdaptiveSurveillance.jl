using Distributions
using Plots
using AdaptiveSurveillance

# Setup
const output_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
const K = 10000 # number of reps
const d = 4 # delay for successful detection
const L = 3
const ν = 1 / 52 # approx once per year there is an outbreak
const Γd = [Geometric(ν) for l = 1:L]
const p0 = 0.01 * ones(L)
const β = 4e-6 * 536 * 7
const p = repeat(prevalance_sequence(p0[1], β), 1, L)
const n = 100
α = 0.1 # go up to 1.0, the higher, the less false positives
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

alarm_counts, atl, p = predictive_value_ratio(1e-3, 52, L, Γd, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1, miniters=Int(1e4))

# Alarm Density
adD = alarm_density(Int(1e4), 52, L, Γd, ones(Int64, L) * typemax(Int64), p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)
adC0 = alarm_density(Int(1e4), 52, L, Γd, zeros(Int64, L), p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)
adC6 = alarm_density(Int(1e4), 52, L, Γd, [26, typemax(Int64), typemax(Int64)], p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)


# successful detection
successful_detections, post_change_alarms = probability_successful_detection(Int(1e4), 4, 1, L, Γd, 10, p0, p, n, apolicy_constant_p, 0.1, tpolicy_constant, 1)

# tf = [true, false]
# for (g1, g2, a1, a2) in Iterators.product(tf, tf, tf, tf)
#     if a1 || a2
#         ta = true
#         if (a1 && !g1) || (a2 && !g2)
#             ta = false
#         end
#         println("$(Int(g1)), $(Int(g2)), $(Int(a1)), $(Int(a2)), $(Int(ta))")
#     end
# end