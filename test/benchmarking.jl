using Test
using Random
using Profile
# using Plots
using ForwardDiff
using BenchmarkTools

using AdaptiveSurveillance

const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")

include("test_data.jl")

ti, tp = 50, 55 # time at prediction, time to predict
Wr, tr = W[1:ti+1], t[1:ti+1] # prevent memory allocation from this showing

# Type stability
# @code_warntype

# Meomory allocation
# AdaptiveSurveillance.future_alarm_log_probability(0, 100, tp, Wr, tr, n)
# Profile.clear_malloc_data()
# AdaptiveSurveillance.future_alarm_log_probability(0, 100, tp, Wr, tr, n)

# Benchmarking
@benchmark AdaptiveSurveillance.future_alarm_log_probability(0, 100, $tp, $Wr, $tr, $n)