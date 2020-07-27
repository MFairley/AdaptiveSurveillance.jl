function plot_sir()
    μ, I, R0, β1, γ1, ρ = [0.01, 0.01, 1.1, 1.3, 0.5, 0.001]
    ph = [μ, I, R0, β1, γ1, ρ]
    sol = prevalence_cost_model_sir(10.0, 110.0, ph)
    plot(sol, labels = ["S" "I" "R" "Cost"], tspan=(0.0,500.0))
end

plot_sir()