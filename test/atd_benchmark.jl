# julia -i atd_benchmark.jl 0.1 0.1 I 50 0.01 0.02 1000 300 1 1

using BenchmarkTools
using AdaptiveSurveillance

includet(joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "result_scripts", "atd_compare.jl"))

# @benchmark run_simulation($2, $astateL)