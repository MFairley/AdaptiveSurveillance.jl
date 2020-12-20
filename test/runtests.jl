using Test
using Random
# using Plots
using ForwardDiff
using BenchmarkTools
using AdaptiveSurveillance

const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "tmp")

include("test_data.jl")

@testset "Logistic Solver" begin
@testset "Verify Gradient and Hessian" begin
    n_checks = 1000
    tΓ = max.(0, t .- Γ_true)
    G = zeros(2)
    H = zeros(2, 2)

    fun = (x) -> AdaptiveSurveillance.log_likelihood(x, tΓ, W, n)
    fun_grad! = (g, x) -> AdaptiveSurveillance.log_likelihood_grad!(g, x, tΓ, W, n)
    fun_hess! = (h, x) -> AdaptiveSurveillance.log_likelihood_hess!(h, x, tΓ, W, n)

    for i = 1:n_checks
        x = rand(2) .* [1, 10] .+ [0, -5]
        # ForwardDiff AutoDiff
        g = x -> ForwardDiff.gradient(fun, x)
        h = x -> ForwardDiff.hessian(fun, x)
        # My gradients
        fun_grad!(G, x)
        fun_hess!(H, x)
        @test all(isapprox.(g(x), G))
        @test all(isapprox.(h(x), H))
    end
end

@testset "Verify Solver" begin
rng = MersenneTwister(1234)
for i = 1:Int(maximum(t))
    for Γ = 0:(i+1)
        if rand(rng) >= 0.01 # do a fraction of the tests for speed
            continue
        end
        objo, βo, zo = AdaptiveSurveillance.solve_logistic_Γ_subproblem_optim(Γ, t[1:i], W[1:i], n)
        objc, βc, zc = AdaptiveSurveillance.solve_logistic_Γ_subproblem_convex(Γ, t[1:i], W[1:i], n)
        @test isapprox(objo, objc, rtol=0.15)
        @test isapprox(βo, βc, rtol=0.15)
        @test isapprox(zo, zc, rtol=0.15)
    end
end
end

# @testset "Profile Likelihood" begin
# ti, tp = 2, 3 # time at prediction, time to predict
# plot_profile_likelihood(0, n, tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 2, 12
# plot_profile_likelihood(0, n, tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 50, 51
# plot_profile_likelihood(0, n, tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 50, 60
# plot_profile_likelihood(0, n, tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 150, 151
# plot_profile_likelihood(0, n, tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# ti, tp = 150, 160
# plot_profile_likelihood(0, n, tp, t[1:ti+1], W[1:ti+1], n, path = save_path)

# end
end