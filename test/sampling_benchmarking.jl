# julia test/benchmarking.jl --track-allocation=user
using Test
using Random
using Profile
# using Plots
using StaticArrays
using StatsBase
using ForwardDiff
using BenchmarkTools

using AdaptiveSurveillance

const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")

include("test_data.jl")

obs = StateObservable(1, n, length(W))
unobs = StateUnobservable(β_true_L, p0_true_L, Γ_true_L)

astateL = AStateLogistic(α, βu, zu)
astateI = AStateIsotonic(α)

for j = 1:100
    AdaptiveSurveillance.update!(j, 1, W[j], obs)
end

# Alarm Funcs
# @benchmark AdaptiveSurveillance.afunc(1, $obs, $astateL)
# @benchmark AdaptiveSurveillance.afunc(1, $obs, $astateI)

# Sampling Policy
tstate_constant = TStateConstant(1)
tstate_random = TStateRandom()
tstate_thompson = TStateThompson(ones(L, 2))
tstate_evsi_clairvoyance = TStateEVSIClairvoyant(unobs)
tstate_evsi = TStateEVSI(βu, zu)
rng_test = MersenneTwister(123)

AdaptiveSurveillance.tfunc(100, obs, astateI, tstate_constant, rng_test)
@benchmark AdaptiveSurveillance.tfunc($100, $obs, $astateI, $tstate_random, $rng_test)
@benchmark AdaptiveSurveillance.tfunc($100, $obs, $astateI, $tstate_thompson, $rng_test)
@benchmark AdaptiveSurveillance.tfunc($100, $obs, $astateI, $tstate_evsi_clairvoyance, $rng_test)
@benchmark AdaptiveSurveillance.tfunc($100, $obs, $astateI, $tstate_evsi, $rng_test) # allocates because of astateI

@benchmark AdaptiveSurveillance.tfunc($100, $obs, $astateL, $tstate_evsi_clairvoyance, $rng_test)
@benchmark AdaptiveSurveillance.tfunc($100, $obs, $astateL, $tstate_evsi, $rng_test)

# Overall Simulation

# ATD



# AdaptiveSurveillance.astat_logistic(t, W, n, βu, zu)

# tp = 101
# Wp = 100
# AdaptiveSurveillance.solve_logistic(tp, Wp, t[1:100], W[1:100], n, βu, zu)