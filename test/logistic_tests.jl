using ForwardDiff
using Random
using AdaptiveSurveillance

@testset "Logistic Solver" begin
@testset "Verify Gradient and Hessian" begin
    n_checks = 1000
    tp, Wp = 301, 20

    fun = (x) -> AdaptiveSurveillance.log_likelihood(x[1], x[2], Γ_true, tp, Wp, t, W, n)
    fun_grad = (x) -> AdaptiveSurveillance.log_likelihood_grad(x[1], x[2], Γ_true, tp, Wp, t, W, n)
    fun_invhess = (x) -> AdaptiveSurveillance.log_likelihood_invhess(x[1], x[2], Γ_true, tp, t, n)

    for i = 1:n_checks
        x = rand(2) .* [1, 10] .+ [0, -5]
        # ForwardDiff AutoDiff
        g = x -> ForwardDiff.gradient(fun, x)
        h = x -> ForwardDiff.hessian(fun, x)

        # My gradient and hessian
        invh11, invh12, invh21, invh22 = fun_invhess(x)
        invH = [invh11 invh12; invh21 invh22]
        
        @test all(isapprox.(g(x), fun_grad(x)))
        @test all(isapprox.(inv(h(x)), invH))
    end
end

# @testset "Verify Solver" begin
# rng = MersenneTwister(1234)
# tp, Wp = 301, 20
# for i = 1:length(t)
#     for Γ = 0:(i-1)
#         if rand(rng) >= 0.01 || (sum(max.(0, t[1:i] .- Γ)) == 0)
#             # do a fraction of the tests for speed
#             # beta value non-identifiable when sum(max.(0, t[1:i] .- Γ)) == 0
#             continue
#         end
#         objo, βo, zo = AdaptiveSurveillance.solve_logistic_Γ_subproblem_optim(Γ, tp, Wp, t[1:i], W[1:i], n)
#         objc, βc, zc = AdaptiveSurveillance.solve_logistic_Γ_subproblem_convex(Γ, vcat(t[1:i], tp), vcat(W[1:i], Wp), n)
#         @test isapprox(objo, objc, rtol=0.15)
#         @test isapprox(βo, βc, atol=1e-2)
#         @test isapprox(zo, zc, atol=1e-2)
#     end
# end
# end

# @testset "Profile Likelihood" begin
# ti, tp = 2, 3 # time at prediction, time to predict
# plot_profile_likelihood(tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 2, 12
# plot_profile_likelihood(tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 50, 51
# plot_profile_likelihood(tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 50, 60
# plot_profile_likelihood(tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 150, 151
# plot_profile_likelihood(tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 150, 160
# plot_profile_likelihood(tp, t[1:ti+1], W[1:ti+1], n, path = save_path)
# end
end