library(here)
library(data.table)
library(ggplot2)
library(isotone)

n = 50
p0 = 0.05
np = 20
p = seq(from = 0.1, to = 1, length.out = np)

# generate data
tau = 4
steps = 30
W = rep(0, steps)
ptrue = rep(0, steps)
for (i in 1:steps) {
  if (i >= tau) {
    W[i] = rbinom(n = 1, size = n, prob = p[min(np, i - tau + 1)])
    ptrue[i] = p[min(np, i - tau + 1)]
  } else {
    W[i] = rbinom(n = 1, size = n, prob = p0)
    ptrue[i] = p0
  }
}

dt <- data.table(t = seq(1, steps), W = W)

ggplot(dt, aes(t, W)) + geom_line()

beta = gpava(y = 2 * asin(sqrt(dt[, W] / n)))$x

# transform beta to probabilities
pbar = (sin (beta / 2))^2

# logit method

