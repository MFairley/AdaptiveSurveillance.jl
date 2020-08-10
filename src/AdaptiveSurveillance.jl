module AdaptiveSurveillance

using DifferentialEquations
using POMDPs

include("sir.jl")
include("pomdp1.jl")

import .pomdp1

export prevalence_cost_model_sir

end # module
