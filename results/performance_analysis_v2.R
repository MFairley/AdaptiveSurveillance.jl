library(here)
library(data.table)
library(survival)
library(survminer)
library(latex2exp)
library(ggplot2)
library(pammtools)
results_path <- here("results", "tmp", "mfairley")
output_path <-  here("results", "tmp")
col_classes <- c("factor", "factor", "factor", "factor", "factor", "integer", "integer", "integer", "integer")
all.files <- list.files(path = results_path, pattern = ".csv", full.names = T)
l <- lapply(all.files, fread, sep=",", colClasses = col_classes)
atd_ind.dt <- rbindlist( l )

# Combined p1p2 variable
atd_ind.dt[, p1p2 := factor(paste0(p1, "_", p2))]

# Set status for survival analysis
atd_ind.dt[, status := 0]
atd_ind.dt[l == 1, status := 1]

# Fit survival curves
sfit <-  survfit(Surv(t, status) ~ g + p1p2 + alg + alarm, data = atd_ind.dt)
sfit.dt <- data.table(surv_summary(sfit, data = atd_ind.dt))[, .(g, p1p2, alg, alarm, time, surv, upper, lower)]

# Publication plot of survival curves
# Add origin points of 1.0
sfit_add.dt <- data.table(CJ(g = unique(sfit.dt$g), p1p2 = unique(sfit.dt$p1p2), alg = unique(sfit.dt$alg), alarm = unique(sfit.dt$alarm), time = 0, surv = 1.0, upper = 1.0, lower = 1.0))
sfit.dt <- rbindlist(list(sfit.dt, sfit_add.dt))[order(g, p1p2, alg, alarm, time)]
sfit.dt[, c("alarm_surv", "alarm_surv_lower", "alarm_surv_upper") := list(1 - surv, 1 - upper, 1 - lower)]

# Factor levels/labels
levels(sfit.dt$g) <- c(TeX("$\\Gamma_1=1$"), TeX("$\\Gamma_1=50$"))
levels(sfit.dt$p1p2) <- c(TeX("$p_l^0 = 0.01, p_2^0 = 0.01$"), TeX("$p_l^0 = 0.01, p_2^0 = 0.02$"), TeX("$p_l^0 = 0.02, p_2^0 = 0.01$"))
levels(sfit.dt$alg) <- c("Clairvoyant", "Profile Likelihood(0.05,0.05)", "Profile Likelihood(0.1,0.1)", "Future Alarm Probability*", "Uniform Random", "Thompson Sampling")
alg_breaks <- c("Clairvoyant", "Future Alarm Probability*", "Profile Likelihood(0.1,0.1)", "Thompson Sampling", "Uniform Random")

vlines.dt <- data.table(g = levels(sfit.dt$g), vline = c(1, 50))
# Logistic
ggplot(sfit.dt[alarm == "logistic" & alg %in% alg_breaks], aes(x = time, y = alarm_surv, ymin=alarm_surv_lower, ymax=alarm_surv_upper, fill=alg, color=alg)) +
  facet_grid(p1p2 ~ g, labeller = label_parsed) + xlim(0, 150) +
  geom_step(na.rm=T) + geom_stepribbon(alpha=0.5, color=NA, show.legend=F) +
  geom_vline(aes(xintercept = vline), data=vlines.dt, color = "red") + 
  xlab("Time (weeks)") + ylab("Cumulative Probability of Alarm in Location 1") + 
  scale_color_discrete(name = "Algorithm", breaks = alg_breaks) + 
  theme_bw() +
  theme(legend.position = "bottom")

# Isotonic
ggplot(sfit.dt[alarm == "isotonic" & alg %in% alg_breaks], aes(x = time, y = alarm_surv, ymin=alarm_surv_lower, ymax=alarm_surv_upper, fill=alg, color=alg)) +
  facet_grid(p1p2 ~ g, labeller = label_parsed) + xlim(0, 150) +
  geom_step(na.rm=T) + geom_stepribbon(alpha=0.5, color=NA, show.legend=F) +
  geom_vline(aes(xintercept = vline), data=vlines.dt, color = "red") + 
  xlab("Time (weeks)") + ylab("Cumulative Probability of Alarm in Location 1") + 
  scale_color_discrete(name = "Algorithm", breaks = alg_breaks) + 
  theme_bw() +
  theme(legend.position = "bottom")
