println("runtests.jl")
using Test
using Plots
println("runtests.jl: packages loaded")
using AdaptiveSurveillance
println("runtests.jl: AdaptiveSurveillance loaded")

include("sir_tests.jl")