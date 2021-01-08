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

astate = AStateLogistic(α, βu, zu)

obs.x .= 1
obs.W .= W

@benchmark AdaptiveSurveillance.afunc(100, obs, astate)

AdaptiveSurveillance.astat_logistic(t, W, n, βu, zu)

tp = 101
Wp = 100
AdaptiveSurveillance.solve_logistic(tp, Wp, t[1:100], W[1:100], n, βu, zu)