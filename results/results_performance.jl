using Random, Distributions
using Plots
using AdaptiveSurveillance

const output_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
const L = 2
const p0 = [0.01, 0.02] # 0.01 * ones(L)
const β = 4e-6 * 536 * 7
const p = repeat(prevalance_sequence(p0[1], β), 1, L) # fix this
const n = 200
const α = 10000 # the higher, the less false positives