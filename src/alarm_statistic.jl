### ALARM STATISTICS
function astat_isotonic(n, positive_counts)
    # @assert all(0 .<= positive_counts .<= n)
    n_visits = length(positive_counts)
    pcon = sum(positive_counts) / (n * n_visits)
    lcon = sum(logpdf(Binomial(n, pcon), positive_counts[i]) for i = 1:n_visits) # to do: use loglikelihood function
    y = 2 * asin.(sqrt.(positive_counts ./ n))
    ir = isotonic_regression!(y)
    piso = sin.(ir ./ 2).^2
    liso = sum(logpdf(Binomial(n, piso[i]), positive_counts[i]) for i = 1:n_visits)
    return liso - lcon, [lcon, liso]
end