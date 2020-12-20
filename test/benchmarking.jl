using Test
using Random
using Profile
# using Plots
using ForwardDiff
using BenchmarkTools

using AdaptiveSurveillance

const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")

include("test_data.jl")

ti, tp = 50, 55 # time at prediction, time to predict
Wr, tr = W[1:ti+1], t[1:ti+1] # prevent memory allocation from this showing
Γr = 25
tΓ = max.(0, tr .- Γr)
x = [β_true, logit(p0_true)]
G = zeros(2)
H = zeros(2, 2)
# Type stability
@code_warntype AdaptiveSurveillance.normalized_log_likelihood(β_true, logit(p0_true), Γr, tr, Wr, n)
@code_warntype AdaptiveSurveillance.log_likelihood(x, tΓ, Wr, n)
@code_warntype AdaptiveSurveillance.log_likelihood_grad!(G, x, tΓ, Wr, n)
@code_warntype AdaptiveSurveillance.log_likelihood_hess!(H, x, tΓ, Wr, n)
@code_warntype AdaptiveSurveillance.log_likelihood_fgh!(G[1], G, H, x, tΓ, Wr, n)
@code_warntype AdaptiveSurveillance.solve_logistic_Γ_subproblem_optim(Γr, tr, Wr, n)

# Meomory allocation
# AdaptiveSurveillance.future_alarm_log_probability(0, 100, tp, Wr, tr, n)
# Profile.clear_malloc_data()
# AdaptiveSurveillance.future_alarm_log_probability(0, 100, tp, Wr, tr, n)

# Benchmarking
@benchmark AdaptiveSurveillance.future_alarm_log_probability(0, 100, $tp, $Wr, $tr, $n)