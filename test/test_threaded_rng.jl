using Random

# function sample(rng)
#     return rand(rng, 1:10)
# end

function thread_samples(K)
    rng = [MersenneTwister(i) for i = 1:Threads.nthreads()]
    samples = zeros(Int64, K)
    Threads.@threads for k = 1:K
        samples[k] = rand(rng[Threads.threadid()], 1:10)
    end
end

thread_samples(100000)