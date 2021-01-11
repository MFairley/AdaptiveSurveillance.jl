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

const base_fn_suffix = "$(Γ_true_L[1])_$(p0_true_L[1])_$(p0_true_L[2])_$(alarm)"

# Simulation Set Up 
const K = 1000 # replications
const maxiters = parse(Int64, ARGS[7])
const run_comparators = parse(Bool, ARGS[8])

println("Running Experiment: βu: $βu, p0u: $p0u, Outbreak Time: $(Γ_true_L[1]), p0: $(p0_true_L), Alarm: $(alarm), Comparators: $(run_comparators)")

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
const astateL = AStateLogistic(α, βu, logit(p0u))
const astateI = AStateIsotonic(α) # to do: change alpha

function write_results(alg_name, atd)
    fn = joinpath(save_path, "atd_$(alg_name)_$(base_fn_suffix).csv")
    write_alarm_time_distribution(obs, unobs, atd, fn)
end

function run_simulation(K, astate)
    if run_comparators
        # Constant / Clairvoyance
        write_results("constant", alarm_time_distribution(K, obs, unobs, astate, TStateConstant(1)))
    
        # Random
        write_results("random", alarm_time_distribution(K, obs, unobs, astate, TStateRandom()))
    
        # Thompson Sampling
        write_results("thompson", alarm_time_distribution(K, obs, unobs, astate, TStateThompson(ones(L, 2))))
        
        # Clairvoyant Future Probability of Alarm
        write_results("evsi_clairvoyant", alarm_time_distribution(K, obs, unobs, astate, TStateEVSIClairvoyant(unobs)))
    end
    write_results("evsi", alarm_time_distribution(K, obs, unobs, astate, TStateEVSI(βu, logit(p0u))))
end

function main()
    if alarm == "L"
        run_simulation(2, astateL)
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