module AdaptiveSurveillance

using DifferentialEquations
using ModelingToolkit

include("sir.jl")

export prevalence_cost_model_sir

end # module
