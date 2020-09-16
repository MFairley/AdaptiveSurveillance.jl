using Random
using Plots
using Distributions
using AdaptiveSurveillance

const output_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
const L = 2
const p0 = [0.01, 0.02]#0.01 * ones(L)
const β = 4e-6 * 536 * 7
const p = repeat(prevalance_sequence(p0[1], β), 1, L)
const n = 200

const ν = 1 / (26) # approx 6 months until there is an outbreak
const Γd = [Geometric(ν) for l = 1:L]

const α = 1000

T_end = 40

# sp2vi, hw2vi = probability_successfull_detection_l(Int(1e4), T_end, 47, 1, L, p0, p, n, astat_isotonic, α,
    # tpolicy_evsi, tstate_evsi(Γd, ones(L, 2), ones(L, 2)));

sp3vi, hw3vi = probability_successfull_detection_l(Int(1e4), T_end, 75, 1, L, p0, p, n, astat_isotonic, α,
    tpolicy_evsi, tstate_evsi(Γd, ones(L, 2), ones(L, 2)));

plot(1:T_end, hcat(sp3vi), label = ["2x" "3x"], xlabel = "Outbreak Start Time", ylabel = "PSD", ylim=(0, 1.0))
savefig(joinpath(output_path, "evsi_psd.png"))