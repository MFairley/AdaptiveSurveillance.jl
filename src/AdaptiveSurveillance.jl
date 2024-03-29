module AdaptiveSurveillance

include("linear_pava.jl")
include("logistic_model.jl")
include("simulation.jl")
include("alarm_statistic.jl")
include("sampling.jl")

export logistic_prevalance
export profile_likelihood
export plot_profile_likelihood

export replication
export average_run_length
export alarm_time_distribution
export write_alarm_time_distribution

export StateObservable
export StateUnobservable

export AStateIsotonic
export AStateLogistic
export AStateLogisticTopr

export TStateConstant
export TStateRandom
export TStateThompson
export TStateEVSI
export TStateEVSIClairvoyant

export calibrate_alarm_threshold

end # module
