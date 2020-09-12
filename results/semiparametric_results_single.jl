using Distributions
using Plots
using AdaptiveSurveillance

# Setup
const output_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
const K = 10000 # number of reps
const d = 4 # delay for successful detection
const ν = 1 / 52 # approx once per year there is an outbreak
const Γd = Geometric(ν)
const p0 = 0.01
const β = 4e-6 * 536 * 7
const p = prevalance_sequence(p0[1], β)
const n = 100
α = 0.1 # go up to 1.0, the higher, the less false positives

# DEBUG
t, false_alarm, delay, test_data, z, thres = sp_single.replication(Γd, Γ, p0, p, n, sp_single.apolicy_isotonic, α)
plot(0:t, )

# PERFORMANCE METRICS
