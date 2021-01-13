# julia test/benchmarking.jl --track-allocation=user
using Test
using Random
using Profile
# using Plots
using StaticArrays
using StatsBase
using ForwardDiff
using BenchmarkTools

using AdaptiveSurveillance

const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")

include("test_data.jl")

ti, tp, Wp = 50, 55, 20 # time at prediction, time to predict
Wr, tr = W[1:ti+1], t[1:ti+1] # prevent memory allocation from this showing
Γr = 25
x = @SVector [β_true, logit(p0_true)]
g = AdaptiveSurveillance.log_likelihood(x, Γr, tp, Wp, tr, Wr, n)
H = AdaptiveSurveillance.log_likelihood_hess(x, Γr, tp, tr, n)

### Type stability
# @code_warntype AdaptiveSurveillance.normalized_log_likelihood(β_true, logit(p0_true), Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.log_likelihood(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.log_likelihood_grad(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.log_likelihood_hess(x, Γr, tp, tr, n)

# @code_warntype AdaptiveSurveillance.activeset(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.is_kkt(x, g)
# @code_warntype AdaptiveSurveillance.is_primal_feasible(x)
# @code_warntype AdaptiveSurveillance.is_dual_feasible(x, g)
# @code_warntype AdaptiveSurveillance.lagrange_multipliers(x, g)
# @code_warntype AdaptiveSurveillance.newtonβ(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.newtonz(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.newtonβz(x, Γr, tp, Wp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.modified_hessian_inv(H)
# @code_warntype AdaptiveSurveillance.convergence_test(x, g)
# @code_warntype AdaptiveSurveillance.solve_logistic_Γ_subproblem(Γr, tp, Wp, tr, Wr, n)

# @code_warntype AdaptiveSurveillance.solve_logistic(tp, Wp, tr, Wr, n, βu, logisitic(p0u))
# @code_warntype AdaptiveSurveillance.profile_log_likelihood(tp, tr, Wr, n)
# @code_warntype AdaptiveSurveillance.profile_likelihood(tp, tr, Wr, n)

### Benchmarking - check memory allocation
# @benchmark AdaptiveSurveillance.normalized_log_likelihood($β_true, $logit(p0_true), $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.log_likelihood($x, $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.log_likelihood_grad($x, $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.log_likelihood_hess($x, $Γr, $tp, $tr, $n)

# @benchmark AdaptiveSurveillance.activeset($x, $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.is_kkt($x, $g)
# @benchmark AdaptiveSurveillance.is_primal_feasible($x)
# @benchmark AdaptiveSurveillance.is_dual_feasible($x, $g)
# @benchmark AdaptiveSurveillance.lagrange_multipliers($x, $g)
# @benchmark AdaptiveSurveillance.newtonβ($x, $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.newtonz($x, $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.newtonβz($x, $Γr, $tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.convergence_test($x, $g)
# @benchmark AdaptiveSurveillance.solve_logistic_Γ_subproblem($Γr, $tp, $Wp, $tr, $Wr, $n)

# @benchmark AdaptiveSurveillance.solve_logistic($tp, $Wp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.profile_log_likelihood($tp, $tr, $Wr, $n)
# @benchmark AdaptiveSurveillance.profile_likelihood($tp, $tr, $Wr, $n, $βu, $zu)

# Meomory allocation tracking
# profile_likelihood(tp, tr, Wr, n)
# Profile.clear_malloc_data()
# profile_likelihood(tp, tr, Wr, n)
