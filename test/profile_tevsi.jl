using Random
using Distributions
using AdaptiveSurveillance

using BenchmarkTools
using Profile

const L = 2
const ν = 1 / (52 * 5) # approx 6 months until there is an outbreak
const Γd = [Geometric(ν) for l = 1:L]
const p0 = [0.01, 0.02]#0.01 * ones(L)
const β = 4e-6 * 536 * 7
const p = repeat(prevalance_sequence(p0[1], β), 1, L)
const n = 200
const α = 1000

Γ = [typemax(Int64), 0]
fraction_forget = 0.5
tstate_t = tstate_thompson(ones(L, 2))
tstate_e = tstate_evsi(Γd, ones(L, 2), fraction_forget)
counts = rand([1.0, 5.0, 2.0, 7.0], 300);
countsp = counts ./ 100

t, la, false_alarm, delay, test_data, locations_visited, ntimes_visisted, last_time_visited, z, w = replication(L, Γ, p0, p, n, 
    astat_isotonic, α, tpolicy_evsi, tstate_e,
    rng1 = MersenneTwister(1), rng2 = MersenneTwister(2), maxiters = 1000);

# @benchmark tpolicy_thompson(L, n, astat_isotonic, α, tstate_t, MersenneTwister(1), test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)

# @benchmark tpolicy_evsi(L, n, astat_isotonic, α, deepcopy(tstate_e), MersenneTwister(1), test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)

# @code_warntype tpolicy_evsi(L, n, astat_isotonic, α, tstate_e, MersenneTwister(1), test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)

function repeat_evsi(K)
    for k = 1:K
        tstate_e = tstate_evsi(Γd, ones(L, 2), fraction_forget)
        tpolicy_evsi(L, n, astat_isotonic, α, tstate_e, MersenneTwister(1), test_data, locations_visited, ntimes_visisted, last_time_visited, z, w, t)
    end
end


@benchmark astat_isotonic($n, $counts, $countsp)

# Profile.clear()
# @profile repeat_evsi(100)
# Profile.print()