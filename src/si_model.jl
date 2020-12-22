using DifferentialEquations

function prevalance_sequence(I0, β; maxt = 10000.0)
    cb = TerminateSteadyState(1e-4, 1e-4, DiffEqCallbacks.allDerivPass)
    u0 = [1 - I0, I0]
    prob = ODEProblem(si_model!, u0, (0.0, maxt), β)
    sol = solve(prob, callback = cb)
    endt = floor(sol.t[end])
    return [sol(t, idxs=2) for t = 0:endt]
end

function si_model!(du, u, p, t)
    β = p
    du[1] = - β * u[1] * u[2]
    du[2] = β * u[1] * u[2]
    nothing
end