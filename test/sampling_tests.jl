using Random, Distributions
# using Plots
using AdaptiveSurveillance

@time res = replication(L, [0 typemax(Int64)], ones(L) * p0_true, p_sequence, n,
    astat_isotonic, α, tpolicy_evsi, tstate_evsi(),
    rng1 = MersenneTwister(1), rng2 = MersenneTwister(2), maxiters = 100);