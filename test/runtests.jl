using Test
using Random
# using Plots
using ForwardDiff
using AdaptiveSurveillance

tmp = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")
if startswith(gethostname(), "sh")
    tmp = ENV["SCRATCH"]
elseif gethostname() != "Michaels-iMac.lan"
    tmp = ""
end
const save_path = tmp

include("test_data.jl")
include("logistic_tests.jl")
include("sampling_tests.jl")
