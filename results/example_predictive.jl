using AdaptiveSurveillance
using CSV, DataFrames

include(joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "test", "test_data.jl"))
const save_path = joinpath(dirname(pathof(AdaptiveSurveillance)), "..", "results", "tmp")

function fill_data(ti, tp)
    data = zeros(n + 1, 4)
    data[:, 1] .= ti
    data[:, 2] .= tp
    data[:, 3] .= 0:n 
    data[:, 4] = profile_likelihood(tp, t[1:ti], W[1:ti], n)
    # mean values
    println(sum(data[:, 3] .* data[:, 4]))
    return data
end

ti, tp = 2, 3 # time at prediction, time to predict
data = fill_data(ti, tp)

ti, tp = 2, 12
data = vcat(data, fill_data(ti, tp))

ti, tp = 50, 51
data = vcat(data, fill_data(ti, tp))

ti, tp = 50, 60
data = vcat(data, fill_data(ti, tp))

ti, tp = 150, 151
data = vcat(data, fill_data(ti, tp))

ti, tp = 150, 160
data = vcat(data, fill_data(ti, tp))

CSV.write(joinpath(save_path, "example_predictive.csv"), DataFrame(data), header = ["ti", "tp", "i", "pl"])