# julia test/benchmarking.jl --track-allocation=user
using Test
using Random
using Profile
# using Plots
using StatsBase
using ForwardDiff
using BenchmarkTools

using AdaptiveSurveillance

const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")

include("test_data.jl")

ti, tp, Wp = 50, 55, 20 # time at prediction, time to predict
Wr, tr = W[1:ti+1], t[1:ti+1] # prevent memory allocation from this showing
Γr = 25
x = [β_true, logit(p0_true)]

# Type stability
# @code_warntype AdaptiveSurveillance.normalized_log_likelihood(β_true, logit(p0_true), Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.log_likelihood(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.log_likelihood_grad(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.log_likelihood_hess(x, Γr, tp, tr, n)
# @code_warntype AdaptiveSurveillance.solve_logistic_Γ_subproblem(Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.solve_logistic(tp, Wp, tr, Wr, n) # fix issue with threads here

# Benchmarking - check memory allocation
# @benchmark AdaptiveSurveillance.log_likelihood_grad($x, $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.log_likelihood_hess($x, $Γr, $tp, $tr, $n)

# @benchmark AdaptiveSurveillance.solve_logistic_Γ_subproblem($Γr, $tp, $Wp, $tr, $Wr, $n)

# @benchmark AdaptiveSurveillance.solve_logistic($Γr, $tp, $Wp, $tr, $Wr, $n)

# Meomory allocation
# profile_likelihood(tp, tr, Wr, n)
# Profile.clear_malloc_data()
#profile_likelihood(tp, tr, Wr, n)

# Benchmarking
# @benchmark profile_likelihood($tp, $tr, $Wr, $n)
