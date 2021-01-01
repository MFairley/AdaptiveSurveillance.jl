library(here)
library(data.table)
library(ggplot2)
results_path <- here("results", "tmp")

example_pred.dt <- fread(paste(results_path, "example_predictive.csv", sep="/"))
example_pred.dt[, tpmti := tp - ti]

# 

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
  theme(legend.position="bottom")
  

ggsave("example_predictive.pdf", width = 4.5, height = 4.5)  
