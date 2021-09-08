# julia atd_compare.jl 0.1 0.1 I 50 0.01 0.02 1000 300 1 1
# Experiment Set Up - Constants
const personal_hostname = "Michaels-MBP.lan"
const L = 5
const r = 2
const lO = L
const β_true = 0.015008
const p0_true = 0.01
const n = 200
const target_arl = 100
const β_true_L = ones(Float64, L) * β_true
const p0_true_L = ones(Float64, L) * p0_true

# Experiment Setup - Variables
const βu = parse(Float64, ARGS[1])
const p0u = parse(Float64, ARGS[2])
const alarm = ARGS[3]
const Γ_lO = parse(Int64, ARGS[4])
p0_true_L[1] = parse(Float64, ARGS[5])
p0_true_L[2] = parse(Float64, ARGS[6])

const base_fn_suffix = "$(Γ_lO)_$(p0_true_L[1])_$(p0_true_L[2])"

# Simulation Set Up 
const K = parse(Int64, ARGS[7])
const maxiters = parse(Int64, ARGS[8])
const run_comparators = parse(Bool, ARGS[9])
const run_evsi = parse(Bool, ARGS[10])

println("Running Experiment: βu: $βu, p0u: $p0u, Alarm: $(alarm), Outbreak Time: $(Γ_lO), p0: $(p0_true_L), Replications: $(K), maxiters: $(maxiters), Comparators: $(run_comparators), EVSI: $(run_evsi)")

using StatsFuns
using DelimitedFiles
using AdaptiveSurveillance

function get_save_path()
    save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")
    sherlock = false
    if startswith(gethostname(), "sh")
        save_path = ENV["SCRATCH"]
        sherlock = true
    elseif gethostname() != personal_hostname
        save_path = ""
    end
    return save_path, sherlock
end

const save_path = get_save_path()[1]
const sherlock = get_save_path()[2]

# System State
const obs = StateObservable(L, n, maxiters)
const unobs = StateUnobservable(β_true_L, p0_true_L, L, lO, Γ_lO)

# Alarm State
const astateI = AStateIsotonic(1.0)
const astateL = AStateLogistic(1.0, βu, p0u)
const astateLr = AStateLogisticTopr(1.0, βu, p0u, r, L)

function run_algorithm(K, astate, tstate, io)
    astate_calibrated, α, arl, hw = calibrate_alarm_threshold(target_arl, obs, unobs, astate, tstate) # calibrate alarm threshold
    writedlm(io, permutedims(vcat(tstate.name, astate.name, α, arl, hw)), ",")
    alarm_time_distribution(K, obs, unobs, astate_calibrated, tstate, save_path)
end

function run_simulation(K, astate)
    fn_env = "$(unobs.Γ[1])_$(unobs.p0[1])_$(unobs.p0[2])"
    calibration_filename = joinpath(save_path, "calibration_$(fn_env).csv")
    open(calibration_filename, "w") do io
        writedlm(io, ["sampling_alg" "alarm_alg" "alpha" "arl" "hw"], ",")
        if run_comparators
            # Constant / Clairvoyance
            run_algorithm(K, astate, TStateConstant(1), io)
        
            # Random
            run_algorithm(K, astate, TStateRandom(), io)
        
            # Thompson Sampling
            run_algorithm(K, astate, TStateThompson(ones(L, 2)), io)
            
            # Clairvoyant Future Probability of Alarm
            run_algorithm(K, astate, TStateEVSIClairvoyant(unobs), io)
        end
        if run_evsi
            run_algorithm(K, astate, TStateEVSI(βu, p0u), io)
        end
    end
end

function main()
    if alarm == "I"
        run_simulation(2, astateI) # precompile
        run_simulation(K, astateI)
    elseif alarm == "L"
        run_simulation(2, astateL)
        run_simulation(K, astateL)
    elseif alarm == "Lr"
        run_simulation(2, astateLr)
        run_simulation(K, astateLr)
    else
        error("Invalid alarm specification")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end