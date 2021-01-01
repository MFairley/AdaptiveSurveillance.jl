using ForwardDiff
using Random
using AdaptiveSurveillance

@testset "Logistic Solver" begin
@testset "Verify Gradient and Hessian" begin
    n_checks = 1000
    tp, Wp = 301, 20

    fun = (x) -> AdaptiveSurveillance.log_likelihood(x, Γ_true, tp, Wp, t, W, n)

    for i = 1:n_checks
        x = rand(2) .* [1, 10] .+ [0, -5]
        # ForwardDiff AutoDiff
        g_fd = x -> ForwardDiff.gradient(fun, x)
        h_fd = x -> ForwardDiff.hessian(fun, x)
        # My gradients
        g = AdaptiveSurveillance.log_likelihood_grad(x, Γ_true, tp, Wp, t, W, n)
        H = AdaptiveSurveillance.log_likelihood_hess(x, Γ_true, tp, t, n)
        @test all(isapprox.(g_fd(x), g))
        @test all(isapprox.(h_fd(x), H))
    end
end

@testset "Bad Conditions" begin
tp, Wp = 301, 20
i = 20
Γ = 20
x0 = @SVector [0.01, 0.0]
x, g = AdaptiveSurveillance.newtonβz(x0, Γ, tp, Wp, t[1:i], Wr[1:i], n)
@test all(isfinite.(x))
end

@testset "Verify Solver" begin
rng = MersenneTwister(1234)
for i = 1:length(t)
    for Γ = 0:(i-1)
        if rand(rng) >= 0.01 || (sum(max.(0, t[1:i] .- Γ)) == 0)
            # do a fraction of the tests for speed
            # beta value non-identifiable when sum(max.(0, t[1:i] .- Γ)) == 0
            continue
        end
        tp = i + rand(rng, 1:10)
        Wp = rand(rng, 0: n)
        @assert(Γ < tp)
        objc, βc, zc = AdaptiveSurveillance.solve_logistic_Γ_subproblem_convex(Γ, vcat(t[1:i], tp), vcat(W[1:i], Wp), n)
        # objo, βo, zo = AdaptiveSurveillance.solve_logistic_Γ_subproblem_optim(Γ, tp, Wp, t[1:i], W[1:i], n)
        # @test isapprox(objo, objc, rtol=0.15)
        # @test isapprox(βo, βc, atol=1e-2)
        # @test isapprox(zo, zc, atol=1e-2)

        # println("i = $i, Γ = $Γ, tp = $tp, Wp = $tp")
        obj, β, z = AdaptiveSurveillance.solve_logistic_Γ_subproblem(Γ, tp, Wp, t[1:i], W[1:i], n)
        @test isapprox(obj, objc, rtol=0.15)
        @test isapprox(β, βc, atol=1e-2)
        @test isapprox(z, zc, atol=1e-2)
    end
end
end

@testset "Profile Likelihood" begin
ti, tp = 2, 3 # time at prediction, time to predict
plot_profile_likelihood(tp, t[1:ti], W[1:ti], n, path = save_path)

ti, tp = 2, 12
plot_profile_likelihood(tp, t[1:ti], W[1:ti], n, path = save_path)

ti, tp = 50, 51
plot_profile_likelihood(tp, t[1:ti], W[1:ti], n, path = save_path)

ti, tp = 50, 60
plot_profile_likelihood(tp, t[1:ti], W[1:ti], n, path = save_path)

ti, tp = 150, 151
plot_profile_likelihood(tp, t[1:ti], W[1:ti], n, path = save_path)

ti, tp = 150, 160
plot_profile_likelihood(tp, t[1:ti], W[1:ti], n, path = save_path)
end
end