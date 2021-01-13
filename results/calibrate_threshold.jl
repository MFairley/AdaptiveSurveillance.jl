using AdaptiveSurveillance

const β_true = 0.015008
const p0_true = 0.01
const n = 200
const L = 1
const β_true_L = ones(Float64, L) * β_true
const p0_true_L = ones(Float64, L) * p0_true
const Γ_true_L = ones(Int64, L) * typemax(Int64)
Γ_true_L[1] = 50
const maxiters = 500
const βu = 0.1
const zu = logit(0.1)

const obs = StateObservable(L, n, maxiters)
const unobs = StateUnobservable(β_true_L, p0_true_L, Γ_true_L)

const α = 1000 
const astateI = AStateIsotonic(α)
const astateL = AStateLogistic(α, βu, zu)

# Sampling Policy
const tstate_constant = TStateConstant(1)

const target = 0.05

println(calibrate_alarm_threshold(target, obs, unobs, astateI, tstate_constant)) # 562.93
println(calibrate_alarm_threshold(target, obs, unobs, astateL, tstate_constant, 2000, 1, 100)) # 11.05
