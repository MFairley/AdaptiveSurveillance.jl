### SEARCH POLICIES
struct tstate_const
    l::Int64
end

function tpolicy_constant(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
    return tstate.l
end

function tpolicy_random(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
    return rand(rng, 1:L) # using rng breaks multi-threading here, to do: file github issue for this
end

struct tstate_thompson
    beta_parameters::Array{Float64, 2}
end

function beta_update(n, beta_parameters, test_data, l, t; accumulate = true)
    if accumulate
        beta_parameters[l, 1] += test_data[t - 1, l]
        beta_parameters[l, 2] += n - test_data[t - 1, l]
    else
        beta_parameters[l, 1] = 1 + test_data[t - 1, l]
        beta_parameters[l, 2] = 1 + n - test_data[t - 1, l]
    end
end

function tpolicy_thompson(L, n, astat, α, tstate, rng, test_data, locations_visited, ntimes_visited, last_time_visited, z, w, t)
    if t > 1
        l = locations_visited[t - 1] # last location visisted
        beta_update(n, tstate.beta_parameters, test_data, l, t)
    end
    d = [Beta(tstate.beta_parameters[l, 1], tstate.beta_parameters[l, 2]) for l = 1:L]
    s = rand.(rng, d)
    l = argmax(s)
    return argmax(s)
end