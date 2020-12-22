using Distributions

const n = 10
const m = 5
const x = 5

function binomail_profile_likelihood(y)
    bc1 = binomial(m, y)
    bc2 = binomial(n, x)
    p = (x + y) / (n + m)
    # d = Binomial(m, p)
    # return pdf(d, x + y)
    return bc2 * bc1 * p^(x+y) * (1 - p)^(n+m-x-y)
end

all_p = [binomail_profile_likelihood(y) for y = 0:m]