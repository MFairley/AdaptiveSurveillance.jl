module pomdp0

using POMDPs, POMDPModelTools, Distributions

struct SurveillancePOMDP <: POMDP{Int64, Symbol, Int64}
    v::Float64 # probability of transitioning to epidemic
    p::Array{Float64} # prevalance sequence
    N::Int64
    n_tests::Int64
    delay_cost::Float64
    false_alarm_cost::Float64
end

# States
POMDPs.states(pomdp::SurveillancePOMDP) = collect(1:pomdp.N) # N is the terminal state
POMDPs.stateindex(pomdp::SurveillancePOMDP, s::Int64) = s
POMDPs.isterminal(pomdp::SurveillancePOMDP, s::Int64) = s == pomdp.N

# Actions
POMDPs.actions(pomdp::SurveillancePOMDP) = [:stop, :continue]

function POMDPs.actionindex(pomdp::SurveillancePOMDP, a::Symbol)
    if a == :stop
        return 1
    elseif a == :continue
        return 2
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
    end
    return Deterministic(pomdp.N - 1)
end

# Observations
POMDPs.observations(pomdp::SurveillancePOMDP) = collect(0:pomdp.n_tests)
POMDPs.obsindex(pomdp::SurveillancePOMDP, o::Int64) = o + 1

function POMDPs.observation(pomdp::SurveillancePOMDP, a::Symbol, s::Int64)
    if s < pomdp.N
        return Binomial(pomdp.n_tests, pomdp.p[s])
    end
    return Deterministic(0)
end

# Rewards
function POMDPs.reward(pomdp::SurveillancePOMDP, s::Int64, a::Symbol)    
    r = 0.0
    if a == :stop && s == 1
        r -= pomdp.false_alarm_cost
    end
    if 1 < s < pomdp.N
        r -= pomdp.delay_cost
    end
    return r
end

# Belief
POMDPs.initialstate(pomdp::SurveillancePOMDP) = Deterministic(1)

POMDPs.discount(pomdp::SurveillancePOMDP) = 1.0

end