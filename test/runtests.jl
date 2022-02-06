using Test
using AdaptiveSurveillance

const personal_hostname = "Michaels-MBP.localdomain"

tmp = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")
if startswith(gethostname(), "sh")
    tmp = ENV["SCRATCH"]
elseif gethostname() != personal_hostname
    tmp = ""
end
const save_path = tmp

include("test_data.jl")
include("logistic_tests.jl")
include("sampling_tests.jl")
