module pomdp1

using POMDPs, POMDPModelTools, Distributions

# const ν = 0.01
# const p = [0.05, 0.1, 0.2, 1.0]
# const N = length(p) + 1
# const n_tests = 20
# const sample_cost = 5.0
# const delay_cost = 50.0

struct SurveillancePOMDP <: POMDP{Bool, Symbol, Bool} # POMDP{State, Action, Observation}
    v::Float64 # probability of transitioning to epidemic
    p::Array{Float64} # prevalance sequence
    N::Int64
    n_tests::Int64
    sample_cost::Float64
    delay_cost::Float64
    false_alarm_cost::Float64
end

# States
POMDPs.states(pomdp::SurveillancePOMDP) = collect(1:pomdp.N) # N is the terminal state
POMDPs.stateindex(pomdp::SurveillancePOMDP, s::Int64) = s

# Actions
POMDPs.actions(pomdp::SurveillancePOMDP) = [:stop, :sample, :continue]

function POMDPs.actionindex(pomdp::SurveillancePOMDP, a::Symbol)
    if a == :stop
        return 1
    elseif a == :sample
        return 2
    elseif a == :continue
        return 3
    end
    error("invalid SurveillancePOMDP action: $a")
end;

# State Transitions
function POMDPs.transition(pomdp::SurveillancePOMDP, s::Int64, a::Symbol)
    if a == :stop || s == pomdp.N
        return Deterministic(pomdp.N)
    elseif s == 1
        return SparseCat([1, 2], [1 - pomdp.v, pomdp.v])
    elseif 1 < s < (pomdp.N - 1)
        return Deterministic(s + 1)
    else
        return Deterministic(pomdp.N - 1)
    end
end

# Observations
POMDPs.observations(pomdp::SurveillancePOMDP) = collect(0:pomdp.n_tests)
POMDPs.obsindex(pomdp::SurveillancePOMDP, o::Int64) = o+1

function POMDPs.observation(pomdp::SurveillancePOMDP, a::Symbol, s::Int64)
    if a == :sample && s < pomdp.N
        return Binomial(pomdp.n_tests, pomdp.p[s])
    else
        return [0]
    end
end

# Rewards
function POMDPs.reward(pomdp::SurveillancePOMDP, s::Int64, a::Symbol)
    if s == pomdp.N
        return 0.0
    end
    
    r = 0.0
    if a == :sample
        r -= pomdp.sample_cost
    elseif a == :stop && s == 1
        r -= pomdp.false_alarm_cost
    end

    if 1 < s < pomdp.N
        r -= pomdp.delay_cost
    end
    
    return r
end

# Belief
POMDPs.initialstate(pomdp::SurveillancePOMDP) = Deterministic(1)

end