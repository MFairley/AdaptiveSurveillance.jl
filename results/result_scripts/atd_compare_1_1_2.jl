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
const L = 5
const β_true = 0.015008
const p0_true = 0.01
const n = 200

const β_true_L = ones(Float64, L) * β_true
const p0_true_L = ones(Float64, L) * p0_true
p0_true_L[2] = 0.02
const Γ_true_L = ones(Int64, L) * typemax(Int64)
Γ_true_L[1] = 1
const maxiters = 150

# System State
obs = StateObservable(L, n, maxiters)
unobs = unobs = StateUnobservable(β_true_L, p0_true_L, Γ_true_L)

# Alarm State
const α = 1000 # the higher, the less false positives
astate = AStateIsotonic(α)

# Sampling Policices
const K = 1000
# Constant
tstate_constant = TStateConstant(1)
println("Starting sampling")
atd_constant = alarm_time_distribution(K, obs, unobs, astate, tstate_constant)
fn = joinpath(save_path, "atd_constant_$(Γ_true_L[1])_$(p0_true_L[1])_$(p0_true_L[2]).csv")
write_alarm_time_distribution(obs, unobs, atd_constant, fn)

# Random
tstate_random = TStateRandom()
atd_random = alarm_time_distribution(K, obs, unobs, astate, tstate_random)
fn = joinpath(save_path, "atd_random_$(Γ_true_L[1])_$(p0_true_L[1])_$(p0_true_L[2]).csv")
write_alarm_time_distribution(obs, unobs, atd_random, fn)

# Thompson Sampling
tstate_thompson = TStateThompson(ones(L, 2))
atd_thompson = alarm_time_distribution(K, obs, unobs, astate, tstate_thompson)
fn = joinpath(save_path, "atd_thompson_$(Γ_true_L[1])_$(p0_true_L[1])_$(p0_true_L[2]).csv")
write_alarm_time_distribution(obs, unobs, atd_thompson, fn)

# Logistic Profile
tstate_evsi = TStateEVSI()
println("Starting evsi")
atd_evsi = alarm_time_distribution(1, obs, unobs, astate, tstate_evsi) # compile
@time atd_evsi = alarm_time_distribution(K, obs, unobs, astate, tstate_evsi) 
fn = joinpath(save_path, "atd_evsi_$(Γ_true_L[1])_$(p0_true_L[1])_$(p0_true_L[2]).csv")
write_alarm_time_distribution(obs, unobs, atd_evsi, fn)