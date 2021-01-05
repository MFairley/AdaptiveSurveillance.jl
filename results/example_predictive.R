library(here)
library(data.table)
library(ggplot2)
library(latex2exp)
results_path <- here("results", "tmp")

example_pred.dt <- fread(paste(results_path, "example_predictive.csv", sep="/"))
example_pred.dt[, tpmti := tp - ti]

tp.labs <- c("Time Horizon = 1", "Time Horizon = 10")
names(tp.labs) <- c(1, 10)
ti.labs <- c("Time = 2", "Time = 50", "Time = 150")
names(ti.labs) <- c(2, 50, 150)

ggplot(example_pred.dt, aes(x=i, y = pl)) +
  facet_grid(ti ~ tpmti, labeller = labeller(ti = ti.labs, tpmti = tp.labs)) +
  geom_col(width=1) +
  xlab("Number of Positive Tests") + ylab("Probability") 

ggsave("example_predictive.pdf", width = 4.5, height = 4.5)  


ggplot(example_pred.dt, aes(x=i, y = pl, fill = factor(tpmti))) +
  facet_grid(vars(ti), labeller = labeller(ti = ti.labs)) +
  geom_col(width=1, alpha = 0.5, position = "identity") +
  xlab("Number of Positive Tests") + ylab("Probability")  +
  scale_fill_discrete(breaks = c(1, 10), name = "Time Horizon") +
  theme_bw() + 
  theme(legend.position="bottom")
  
ggsave("example_predictive.pdf", width = 6.5, height = 6.5)  

# data for predictive distributions
W <- c(1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0, 0.0, 4.0, 3.0, 2.0, 2.0, 
       2.0, 4.0, 2.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 8.0, 2.0, 5.0, 
       3.0, 0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 
       2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 6.0, 2.0, 2.0, 1.0, 2.0, 1.0, 4.0, 2.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 
       2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 5.0, 3.0, 0.0, 2.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 3.0, 
       3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 4.0, 1.0, 1.0, 3.0, 1.0, 2.0, 4.0, 2.0, 1.0, 1.0, 
       4.0, 1.0, 0.0, 2.0, 1.0, 5.0, 2.0, 3.0, 3.0, 4.0, 0.0, 6.0, 4.0, 3.0, 4.0, 3.0, 1.0, 4.0, 5.0, 5.0, 
       1.0, 2.0, 5.0, 4.0, 4.0, 4.0, 7.0, 3.0, 2.0, 2.0, 7.0, 5.0, 3.0, 2.0, 3.0, 4.0, 4.0, 3.0, 6.0, 1.0, 
       6.0, 4.0, 1.0, 4.0, 8.0, 2.0, 2.0, 3.0, 3.0, 2.0, 4.0, 3.0, 8.0, 2.0, 7.0, 7.0, 5.0, 8.0, 9.0, 6.0, 
       8.0, 8.0, 4.0, 5.0, 8.0, 2.0, 8.0, 4.0, 6.0, 6.0, 13.0, 9.0, 12.0, 8.0, 6.0, 8.0, 4.0, 5.0, 9.0, 4.0, 
       9.0, 8.0, 4.0, 7.0, 10.0, 11.0, 12.0, 8.0, 11.0, 9.0, 12.0, 12.0, 13.0, 13.0, 10.0, 10.0, 11.0, 15.0, 
       12.0, 11.0, 4.0, 7.0, 7.0, 9.0, 13.0, 11.0, 13.0, 16.0, 11.0, 9.0, 16.0, 12.0, 11.0, 15.0, 11.0, 9.0, 
       16.0, 11.0, 13.0, 16.0, 12.0, 17.0, 11.0, 20.0, 16.0, 16.0, 19.0, 15.0, 12.0, 13.0, 12.0, 14.0, 15.0, 
       17.0, 22.0, 20.0, 16.0, 20.0, 17.0, 18.0, 17.0, 19.0, 16.0, 19.0, 17.0, 23.0, 24.0, 22.0, 19.0, 19.0, 
       20.0, 19.0, 23.0, 28.0, 20.0, 20.0, 24.0, 26.0, 25.0, 26.0, 21.0, 34.0, 29.0, 28.0, 23.0, 29.0, 28.0, 
       27.0, 36.0, 34.0, 29.0, 22.0, 17.0, 29.0, 28.0, 23.0, 39.0, 20.0, 28.0, 31.0, 23.0, 37.0, 31.0, 39.0, 
       49.0)
sample_data.dt <- data.table(W = W, t = 1:length(W))

ggplot(sample_data.dt, aes(x = t, y = W)) + geom_line() +
  xlab("Time (weeks)") + ylab(TeX("Positive count ($W_t$) out of n = 200")) +
  scale_x_continuous(limits = c(1, length(W)), breaks = c(1, 50, 100, 150, 200, 250, 300)) +
  geom_vline(xintercept = 100, color = "red") + 
  annotate(geom="text", label=TeX("$\\Gamma_l=100$", output='character'), x = 120, y = 30, parse=TRUE) +
  theme_bw()
ggsave("example_data.pdf", width = 6.5, height = 6.5)  










