using Distributions
using AdaptiveSurveillance

const output_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
const K = 1000 # number of reps
const L = 3
const ν = 1 / 52 # approx once per year there is an outbreak
const Γd = [Geometric(ν) for l = 1:L]
const p0 = 0.01
const β = 4e-6 * 536 * 7
const p = prevalance_sequence(p0, β)
const n = 100
α = 0.1

apolicy_constant_p(L, Γd, n, α, test_data, t) = apolicy_constant(L, Γd, n, α, test_data, t, apolicy_isotonic, 1)

# Average Run Length 0
arl0_constant = arl(0, K, L, Γd, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)
arl1_constant = arl(1, K, L, Γd, p0, p, n, apolicy_constant_p, α, tpolicy_constant, 1)

# Simulate best case performance by sampling only one location
apolicy_constant_p(L, Γd, n, α, test_data, t) = apolicy_constant(L, Γd, n, α, test_data, t, AdaptiveSurveillance.apolicy_isotonic, 1)
# simulation(K, joinpath(output_path, "bound_l1.csv"), L, Γd, [], p0, p, n, apolicy_constant_p, α, AdaptiveSurveillance.tpolicy_constant, 1)