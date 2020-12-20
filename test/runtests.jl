using Test
using Random
# using Plots
using ForwardDiff
using BenchmarkTools
using AdaptiveSurveillance

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
for i = 1:maximum(t)
    for Γ = 0:(i+1)
        if rand(rng) >= 0.001 # do a fraction of the tests for speed
            continue
        end
        objo, βo, zo = AdaptiveSurveillance.solve_logistic_Γ_subproblem_optim(Γ, t, W, n)
        objc, βc, zc = AdaptiveSurveillance.solve_logistic_Γ_subproblem_convex(Γ, t, W, n)
        @test isapprox(objo, objc, rtol=0.15)
        @test isapprox(βo, βc, rtol=0.15)
        @test isapprox(zo, zc, rtol=0.15)
    end
end
end

# res1 = solve_logistic_optim(W, t, Γ_true, n)

# lp = profile_log_likelihood(0, 100, 301, W, t, n)
# println(res1)
# @time res2 = solve_logistic_convex(W, t, Γ_true, n)
# println(res2)

# softmax(y_likeli)
end