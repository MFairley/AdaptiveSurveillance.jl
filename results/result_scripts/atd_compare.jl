# Experiment Set Up - Constants
const L = 5
const β_true = 0.015008
const p0_true = 0.01
const n = 200
const β_true_L = ones(Float64, L) * β_true
const p0_true_L = ones(Float64, L) * p0_true
const Γ_true_L = ones(Int64, L) * typemax(Int64)

const α = 1000 # alarm threshold, the higher, the less false positives

# Experiment Setup - Variables
const βu = parse(Float64, ARGS[1])
const p0u = parse(Float64, ARGS[2])
Γ_true_L[1] = parse(Int64, ARGS[3])
p0_true_L[1] = parse(Float64, ARGS[4])
p0_true_L[2] = parse(Float64, ARGS[5])
const alarm = ARGS[6]

const base_fn_suffix = "$(Γ_true_L[1])_$(p0_true_L[1])_$(p0_true_L[2])"

# Simulation Set Up 
const K = 1000 # replications
const maxiters = parse(Int64, ARGS[7])
const run_comparators = parse(Bool, ARGS[8])

if run_comparators
    println("Running Experiment: βu: $βu, p0u: $p0u, Outbreak Time: $(Γ_true_L[1]), p0: $(p0_true_L) WITH comparators")
else
    println("Running Experiment: βu: $βu, p0u: $p0u, Outbreak Time: $(Γ_true_L[1]), p0: $(p0_true_L) WITHOUT comparators")
end

using StatsFuns
using AdaptiveSurveillance

function get_save_path()
    save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
    sherlock = false
    if startswith(gethostname(), "sh")
        save_path = ENV["SCRATCH"]
        sherlock = true
    elseif gethostname() != "Michaels-iMac.lan"
        save_path = ""
    end
    return save_path, sherlock
end

const save_path = get_save_path()[1]
const sherlock = get_save_path()[2]

# System State
obs = StateObservable(L, n, maxiters)
unobs = unobs = StateUnobservable(β_true_L, p0_true_L, Γ_true_L)

# Alarm State
astate = AStateIsotonic(α)
if alarm == "L"
    astate = AStateLogistic(α, βu, logit(p0u))
end

# Sampling Policices
if run_comparators
    # Constant / Clairvoyance
    tstate_constant = TStateConstant(1)
    atd_constant = alarm_time_distribution(K, obs, unobs, astate, tstate_constant)
    fn = joinpath(save_path, "atd_constant_$(base_fn_suffix).csv")
    write_alarm_time_distribution(obs, unobs, atd_constant, fn)

    # Random
    tstate_random = TStateRandom()
    atd_random = alarm_time_distribution(K, obs, unobs, astate, tstate_random)
    fn = joinpath(save_path, "atd_random_$(base_fn_suffix).csv")
    write_alarm_time_distribution(obs, unobs, atd_random, fn)

    # Thompson Sampling
    tstate_thompson = TStateThompson(ones(L, 2))
    atd_thompson = alarm_time_distribution(K, obs, unobs, astate, tstate_thompson)
    fn = joinpath(save_path, "atd_thompson_$(base_fn_suffix).csv")
    write_alarm_time_distribution(obs, unobs, atd_thompson, fn)

    # Clairvoyant Future Probability of Alarm
    tstate_evsi_clairvoyance = TStateEVSIClairvoyant(unobs)
    atd_evsi_clairvoyant = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi_clairvoyance)
    fn = joinpath(save_path, "atd_evsi_clairvoyant_$(base_fn_suffix).csv")
    write_alarm_time_distribution(obs, unobs, atd_evsi_clairvoyant, fn)
end

# Logistic Profile Likelihood
println("Starting Profile Likelihood")
tstate_evsi = TStateEVSI(βu, logit(p0u))
atd_evsi = alarm_time_distribution(1, obs, unobs, astate, tstate_evsi) # compile
@time atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi) # run
fn = joinpath(save_path, "atd_evsi_$(βu)_$(p0u)_$(base_fn_suffix).csv")
write_alarm_time_distribution(obs, unobs, atd_evsi, fn)