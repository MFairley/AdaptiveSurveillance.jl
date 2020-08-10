module pomdp1

using POMDPs, POMDPModelTools

const L = 2
const p0 = 0.01
const p1 = 0.1

struct SurveillancePOMDP <: POMDP{Bool, Symbol, Bool} # POMDP{State, Action, Observation}
    L::Int64 # number of locations
    v::Float64 # probability of transitioning to epidemic
    p0::Float64 # endemic prevalance
    p1::Float64 # epidemic prevalance
end

sPOMDP = SurveillancePOMDP(L, 0.001, p0, p1)

# states
state_i_dict = Dict(
    (0, 0) => 1,
    (1, 0) => 2,
    (2, 0) => 3,
    (0, 1) => 4,
    (1, 1) => 5,
    (2, 1) => 6,
    (0, 2) => 7,
    (1, 2) => 8,
    (2, 2) => 9
)

POMDPs.states(pomdp::SurveillancePOMDP) = vec(collect(Iterators.product(ntuple(i->[0, 1, 2], L)...))))
POMDPs.stateindex(pomdp::SurveillancePOMDP, s::Tuple) = state_i_dict[s]

# actions
tmp_act = vec(collect(Iterators.product(ntuple(i->[0, 1, 2], L)...)))
to_del = []
for (i, a) in enumerate(tmp_act)
    if sum(a .== 2) > 1
        push!(to_del, i)
        
    end
end
deleteat!(tmp_act, to_del)

POMDPs.actions(pomdp::SurveillancePOMDP) = tmp_act

act_i_dict = Dict(
    (0, 0) => 1,
    (1, 0) => 2,
    (2, 0) => 3,
    (0, 1) => 4,
    (1, 1) => 5,
    (2, 1) => 6,
    (0, 2) => 7,
    (1, 2) => 8
)

POMDPs.actionindex(pomdp::TigerPOMDP, a::Tuple) = act_i_dict

# state transitions
function POMDPs.transition(pomdp::SurveillancePOMDP, s::Tuple, a::Tuple)
    # returns a function that samples next state and has pmf

end


end