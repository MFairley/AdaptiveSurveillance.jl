function prevalence_cost_model_sir(τ::Float64, r::Float64, ph::Array{Float64}; max_add_t=10000)
    @assert τ <= r
    μ, I, R0, β1, γ1, ρ = ph
    β0 = μ * (R0 - 1) / I
    γ0 = β0 / R0
    S = 1 / R0
    R = 1 - S - I
    change_times = [τ, r]
    affect!(integrator) = begin
        β = β1
        γ = γ1
        if integrator.t >= r
            β = β0
            γ = γ0
        end
        integrator.p[2] = β
        integrator.p[3] = γ
        
    end
    termination(integrator, abstol, reltol) = begin
        DiffEqCallbacks.allDerivPass(integrator, abstol, reltol) & (integrator.t >= r)
    end
    u0 = [S, I, R, 0.0]
    @assert sum(u0) == 1.0
    tspan = (0, r+max_add_t)
    p = [μ, β0, γ0, ρ]
    prob = ODEProblem(sir_model!, u0, tspan, p)
    # prob = sir_model_mt(u0, tspan, p)
    cb1 = PresetTimeCallback(change_times, affect!)
    cb2 = TerminateSteadyState(1e-8, 1e-6, termination)
    solve(prob, callback = CallbackSet(cb1, cb2))
end

function sir_model!(du, u, p, t)
    μ, β, γ, ρ = p
    du[1] = μ - β*u[1]*u[2] - μ*u[1]
    du[2] = β*u[1]*u[2] - γ*u[2] - μ*u[2]
    du[3] = γ*u[2] - μ*u[3]
    du[4] = exp(- t * ρ) * u[2] # discounted infection time
    nothing
end

function sir_model_mt(u0, tspan, p)
    @parameters t, μ, β, γ, ρ
    @variables S(t) I(t) R(t) c(t)
    @derivatives D'~t
    eqs = [
        D(S) ~ μ - β*S*I - μ*S,
        D(I) ~ β*S*I - γ*I - μ*I,
        D(R) ~ γ*I - μ*R,
        D(c) ~ exp(-t * ρ) * I
    ]
    sys = ODESystem(eqs)
    u0 = [
        S => u0[1]
        I => u0[2]
        R => u0[3]
        c => u0[4]
    ]
    p = [
        μ => p[1]
        β => p[2]
        γ => p[3]
        ρ => p[4]
    ]
    prob = ODEProblem(sys, u0, tspan, p)
end