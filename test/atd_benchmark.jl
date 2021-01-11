# julia -i atd_benchmark.jl 0.1 0.1 1 0.01 0.02 I 30000 1

using BenchmarkTools
using AdaptiveSurveillance

include(joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "result_scripts", "atd_compare.jl"))

# @benchmark run_simulation(2, astateL)