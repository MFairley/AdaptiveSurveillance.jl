println("Hello, World")
println("There are $(Threads.nthreads()) threads")
using StatsFuns
using AdaptiveSurveillance
println("Module Loaded")

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

# Problem Set Up
const L = 2
const β_true = 0.015008
const p0_true = 0.01
const n = 200
const maxiters = 150
const Γ_true = 1
const Γ = ones(Int64, L) * typemax(Int64)
Γ[1] = Γ_true

# System State
obs = StateObservable(L, n, maxiters)
unobs = StateUnobservable(Γ, (t, Γ) -> logistic_prevalance(β_true, logit(p0_true), Γ, t))

# Alarm State
const α = 10000 # the higher, the less false positives
astate = AStateIsotonic(α)

# Sampling Policices
const K = 2
# Constant
tstate_constant = TStateConstant(1)
println("Starting sampling")
atd_constant = alarm_time_distribution(K, obs, unobs, astate, tstate_constant)
write_alarm_time_distribution(obs, unobs, atd_constant, joinpath(save_path, "atd_constant_$(Γ_true).csv"))

# Random
tstate_random = TStateRandom()
atd_random = alarm_time_distribution(K, obs, unobs, astate, tstate_random)
write_alarm_time_distribution(obs, unobs, atd_random, joinpath(save_path, "atd_random_$(Γ_true).csv"))

# Thompson Sampling
tstate_thompson = TStateThompson(ones(L, 2))
atd_thompson = alarm_time_distribution(K, obs, unobs, astate, tstate_thompson)
write_alarm_time_distribution(obs, unobs, atd_thompson, joinpath(save_path, "atd_thompson_$(Γ_true).csv"))

# Logistic Profile
tstate_evsi = TStateEVSI()
println("Starting evsi")
@time atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi)
write_alarm_time_distribution(obs, unobs, atd_evsi, joinpath(save_path, "atd_evsi_$(Γ_true).csv"))