library(here)
library(data.table)
library(survival)
library(survminer)
library(latex2exp)
library(ggplot2)
library(pammtools)
library(scales)
results_path <- here("results", "tmp", "mfairley")
output_path <-  here("results", "tmp")
col_classes <- c("factor", "factor", "factor", "factor", "factor", "integer", "integer", "integer", "integer")
all.files <- list.files(path = results_path, pattern = ".csv", full.names = T)
l <- lapply(all.files, fread, sep=",", colClasses = col_classes)
atd_ind.dt <- rbindlist( l )

# Combined p1p2 variable
atd_ind.dt[, p1p2 := factor(paste0(p1, "_", p2), levels = c("0.01_0.01", "0.01_0.02", "0.02_0.01"))]

# Re-order and relabel factors
atd_ind.dt[, alg := factor(alg, levels = c("constant", "evsi_clairvoyant", "evsi_0.05_0.05", "evsi_0.1_0.1", "thompson", "random"),
                           labels = c("Clairvoyant", "Future Alarm Pr*", "Profile Likelihood(0.05,0.05)", "Profile Likelihood", "Thompson Sampling", "Uniform Random"))]

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

# LaTeX Factor levels/labels
levels(sfit.dt$g) <- c(TeX("$\\Gamma_1=1$"), TeX("$\\Gamma_1=50$"))
levels(sfit.dt$p1p2) <- c(TeX("$p_l^0 = 0.01, p_2^0 = 0.01$"), TeX("$p_l^0 = 0.01, p_2^0 = 0.02$"), TeX("$p_l^0 = 0.02, p_2^0 = 0.01$"))
alg_breaks <- c("Clairvoyant", "Future Alarm Pr*", "Profile Likelihood(0.05,0.05)", "Profile Likelihood", "Thompson Sampling", "Uniform Random")

vlines.dt <- data.table(g = levels(sfit.dt$g), vline = c(1, 50))
# Logistic
ggplot(sfit.dt[alarm == "logistic" & alg %in% alg_breaks], aes(x = time, y = alarm_surv, ymin=alarm_surv_lower, ymax=alarm_surv_upper, fill=alg, color=alg)) +
  facet_grid(p1p2 ~ g, labeller = label_parsed) + xlim(0, 150) +
  geom_step(na.rm=T) + geom_stepribbon(alpha=0.5, color=NA, show.legend=F) +
  geom_vline(aes(xintercept = vline), data=vlines.dt, color = "red") + 
  xlab("Time (weeks)") + ylab("Cumulative Probability of Alarm in Location 1") + 
  scale_color_discrete(name = "Algorithm", breaks = alg_breaks) + 
  theme_bw() +
  theme(legend.position = "bottom") +
  guides(colour=guide_legend(ncol=3,nrow=2,byrow=TRUE))
ggsave(paste(output_path, "survival_curves.pdf", sep="/"), width=6.5, height=6.5)

# Isotonic
ggplot(sfit.dt[alarm == "isotonic" & alg %in% alg_breaks], aes(x = time, y = alarm_surv, ymin=alarm_surv_lower, ymax=alarm_surv_upper, fill=alg, color=alg)) +
  facet_grid(p1p2 ~ g, labeller = label_parsed) + xlim(0, 150) +
  geom_step(na.rm=T) + geom_stepribbon(alpha=0.5, color=NA, show.legend=F) +
  geom_vline(aes(xintercept = vline), data=vlines.dt, color = "red") + 
  xlab("Time (weeks)") + ylab("Cumulative Probability of Alarm in Location 1") + 
  scale_color_discrete(name = "Algorithm", breaks = alg_breaks) + 
  theme_bw() +
  theme(legend.position = "bottom") +
  guides(fill=guide_legend(ncol=1,nrow=2,byrow=TRUE))

# False Alarm Probability
fap.dt <- atd_ind.dt[l > 0, .(fa = sum(false_alarm == 1), nfa = sum(false_alarm == 0), n = .N), by=.(p1p2, g, alg, alarm)]
z <- qnorm(0.975)
fap.dt[, p := fa / n]
fap.dt[, c("p_lower", "p_upper") := .(sin(asin(sqrt(p)) - z / (2 * sqrt(n)))^2, sin(asin(sqrt(p)) + z / (2 * sqrt(n)))^2)]
fap.dt[, fprob_fmt := paste0(comma(p, accuracy = 0.01), " [", comma(p_lower, accuracy = 0.01), ", ", comma(p_upper, accuracy = 0.01), "]")]

# Median Conditional Delay
med_cond_delay <- function(g_sel) {
  sfit1 <-  survfit(Surv(t, status) ~ g + p1p2 + alg + alarm, data = atd_ind.dt[g == g_sel], start.time = g_sel - 1)
  sfit1.dt <- unique(data.table(surv_summary(sfit1, data = atd_ind.dt))[, .(g, p1p2, alg, alarm, strata)])
  med1.dt <- data.table(surv_median(sfit1))
  med1.dt[, strata := factor(strata, levels = levels(sfit1.dt$strata))]
  comb.dt <- merge(sfit1.dt, med1.dt, by = "strata")[, .(g, p1p2, alg, alarm, median, lower, upper)]
  comb.dt[, c("median", "lower", "upper") := .(median - g_sel, lower - g_sel, upper - g_sel )]
  return(comb.dt)
}

delay1.dt <- med_cond_delay(1)
delay50.dt <- med_cond_delay(50)
delay.dt <- rbindlist(list(delay1.dt, delay50.dt))
delay.dt[, med_fmt := paste0(comma(median, accuracy = 1.0), " [", comma(lower, accuracy = 1.0), ", ", comma(upper, accuracy = 1.0), "]")]

tor.dt <- merge(delay.dt[, .(p1p2, g, alg, alarm, med_fmt)], fap.dt[, .(p1p2, g, alg, alarm, fprob_fmt)], by = c("p1p2", "g", "alg", "alarm"))
tor.dt <-  dcast(tor.dt, alarm + p1p2 + alg ~ g, value.var = c("fprob_fmt", "med_fmt"))
fwrite(tor.dt[alg %in% alg_breaks & alarm == "logistic"], paste(output_path, "table_of_results_raw.csv", sep="/"))




