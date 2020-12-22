using Test
using Random
# using Plots
using ForwardDiff
using AdaptiveSurveillance

const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")

include("test_data.jl")
# include("logistic_tests.jl")
include("sampling_policy_tests.jl")
