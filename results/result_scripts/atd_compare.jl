# julia atd_compare.jl 0.1 0.1 I 50 0.01 0.02 1000 300 1 1
# Experiment Set Up - Constants
const L = 5
const β_true = 0.015008
const p0_true = 0.01
const n = 200
const β_true_L = ones(Float64, L) * β_true
const p0_true_L = ones(Float64, L) * p0_true
const Γ_true_L = ones(Int64, L) * typemax(Int64)

const αL = 11.05 # calibrated to 0.05 fa for constant for gamma = 50
const αI = 562.93 # alarm threshold, the higher, the less false positives

# Experiment Setup - Variables
const βu = parse(Float64, ARGS[1])
const p0u = parse(Float64, ARGS[2])
const alarm = ARGS[3]
Γ_true_L[1] = parse(Int64, ARGS[4])
p0_true_L[1] = parse(Float64, ARGS[5])
p0_true_L[2] = parse(Float64, ARGS[6])

const base_fn_suffix = "$(Γ_true_L[1])_$(p0_true_L[1])_$(p0_true_L[2])"

# Simulation Set Up 
const K = parse(Int64, ARGS[7])
const maxiters = parse(Int64, ARGS[8])
const run_comparators = parse(Bool, ARGS[9])
const run_evsi = parse(Bool, ARGS[10])

println("Running Experiment: βu: $βu, p0u: $p0u, Alarm: $(alarm), Outbreak Time: $(Γ_true_L[1]), p0: $(p0_true_L), Replications: $(K), maxiters: $(maxiters), Comparators: $(run_comparators), EVSI: $(run_evsi)")

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
const obs = StateObservable(L, n, maxiters)
const unobs = StateUnobservable(β_true_L, p0_true_L, Γ_true_L)

# Alarm State
const astateL = AStateLogistic(αL, βu, p0u)
const astateI = AStateIsotonic(αI)

function run_simulation(K, astate)
    if run_comparators
        # Constant / Clairvoyance
        alarm_time_distribution(K, obs, unobs, astate, TStateConstant(1), save_path)
    
        # Random
        alarm_time_distribution(K, obs, unobs, astate, TStateRandom(), save_path)
    
        # Thompson Sampling
        alarm_time_distribution(K, obs, unobs, astate, TStateThompson(ones(L, 2)), save_path)
        
        # Clairvoyant Future Probability of Alarm
        alarm_time_distribution(K, obs, unobs, astate, TStateEVSIClairvoyant(unobs), save_path)
    end
    if run_evsi
        alarm_time_distribution(K, obs, unobs, astate, TStateEVSI(βu, p0u), save_path)
    end
end

function main()
    if alarm == "L"
        run_simulation(2, astateL) # precompile
        run_simulation(K, astateL)
    elseif alarm == "I"
        run_simulation(2, astateI)
        run_simulation(K, astateI)
    else
        error("Invalid alarm specification")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end