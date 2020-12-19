module AdaptiveSurveillance

include("sir.jl")
include("logistic_solver.jl")
include("semiparametric_single.jl")
include("semiparametric.jl")

import .sp_single

# export prevalence_cost_model_sir
export sp_single

export prevalance_sequence
export simulation
export tpolicy_constant
export apolicy_isotonic
export apolicy_constant
export arl
export replication
export fixedΓ_alarm_distribution
export predictive_value
export predictive_value_ratio
export alarm_density
export probability_successful_detection
export astat_isotonic

export alarm_time_distribution
export probability_successfull_detection_l
export tpolicy_random
export tpolicy_thompson
export tpolicy_evsi

export tstate_thompson
export tstate_const

export tstate_evsi

export solve_logistic_optim
export solve_logistic_convex
end # module
