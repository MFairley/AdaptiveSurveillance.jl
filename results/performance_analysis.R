library(here)
library(data.table)
library(survival)
library(survminer)
library(pammtools)
library(ggplot2)
library(scales)
library(stringr)
library(latex2exp)
results_path <- here("results", "tmp", "mfairley")
output_path <-  here("results", "tmp")

# Experimental Setup
# Algorithms
algs = c("constant", "evsi_0.1_0.1", "thompson", "random")
alg_labels <- c("Clairvoyance", "Profile Likelihood", "Thompson Sampling", "Uniform Random")
alarms <- c("I", "L")
alarm_labels <- c("Isotonic", "Logistic")
# Environments
gammas <- c(1, 50)
p1s <- c(0.01, 0.02)
p2s <- c(0.02, 0.01)
p1p2_levels <- c("0.01_0.01", "0.01_0.02", "0.02_0.01")

L <- 5
header <- c(sprintf("p%d",seq(1:L)), sprintf("a%d",seq(from = 0, to = L)))

# Scenario / Experiment: g, p1, p2, alg
read_scenario_alg <- function(g, p1, p2, alg, alarm) {
  atd_alg.dt <- fread(paste(results_path, paste0("atd_", alg, "_", g, "_", p1, "_", p2, "_", alarm,".csv"), sep="/"), col.names = header)
  atd_alg.dt[, t := 1:.N]
  atd_alg.dt[, g := factor(g)]
  atd_alg.dt[, p1p2 := factor(paste0(p1, "_", p2), levels = p1p2_levels)]
  atd_alg.dt[, alg :=  factor(alg, levels = algs, labels = alg_labels)]
  atd_alg.dt[, alarm := factor(alarm, levels = alarms, labels = alarm_labels)]
  return(atd_alg.dt)
}

read_scenario <- function(g, p1, p2, alarm) {
  atd.dt <- data.table()
  for (a in algs) {
    tmp.dt <- read_scenario_alg(g, p1, p2, a, alarm)
    atd.dt <- rbindlist(list(atd.dt, tmp.dt), use.names = T)
  }
  atd.dt[, alg := factor(alg, levels = algs, labels = alg_labels)]
  atd.dt[, alarm := factor(alarm, levels = alarms, labels = alarm_labels)]
  return(atd.dt)
}

read_scenario_alg_ind <- function (g, p1, p2, alg, alarm) { 
  atd.dt <- read_scenario_alg(g, p1, p2, alg, alarm)
  a0s <- atd.dt[, sum(a0)]
  if (a0s > 0) {
    stop("a0 > 0")
  }
  # assuming outbreak in location 1
  ind.dt <- data.table(t = rep(atd.dt$t, atd.dt$a1), status = 1)[!is.na(t)]
  for (i in 2:L) {
    if (sum(atd.dt[, paste0("a", i), with=F]) > 0) {
      tmp.dt <- data.table(t = rep(atd.dt$t, unlist(atd.dt[, paste0("a", i), with=F])), status = 0)[!is.na(t)]
      ind.dt <- rbindlist(list(ind.dt, tmp.dt), use.names = T)
    }
  }
  ind.dt[, g := factor(g)]
  ind.dt[, p1p2 := factor(paste0(p1, "_", p2), levels = p1p2_levels)]
  ind.dt[, alg := factor(alg, levels = algs, labels = alg_labels)]
  ind.dt[, alarm := factor(alarm, levels = alarms, labels = alarm_labels)]
  return(ind.dt)
}

read_scenario_individual <- function(g, p1, p2, alarm) {
  atd.dt <- data.table()
  for (a in algs) {
    tmp.dt <- read_scenario_alg_ind(g, p1, p2, a, alarm)
    atd.dt <- rbindlist(list(atd.dt, tmp.dt), use.names = T)
  }
  atd.dt[, alg := factor(alg, levels = algs, labels = alg_labels)]
  atd.dt[, alarm := factor(alarm, levels = alarms, labels = alarm_labels)]
  return(atd.dt)
}

# False Alarm Probability (for all locations)any location)
false_alarm_prob <- function(atd_ind.dt, acc=0.01) {
  fap.dt <- atd_ind.dt[, .(fa=sum((status == 1) & (t < as.numeric(as.character(g)))) + sum(status == 0), n=.N), by = .(p1p2, alg, g)]
  fap.dt[, p := fa / n]
  fap.dt[, hw := sqrt(p * (1-p) / n)]
  fap.dt[, lower := p - hw]
  fap.dt[, upper := p + hw]
  fap.dt[, fprob_fmt := paste0(comma(p, accuracy = acc), " [", comma(lower, accuracy = acc), ", ", comma(upper, accuracy = acc), "]")]
  return(fap.dt)
}

# Median Conditional Delay
median_cond_delay <- function(atd_ind.dt, g_sel, acc=1.0) {
  sfit_delay <-  survfit(Surv(t, status) ~ alg + g + p1p2, data = atd_ind.dt[g == g_sel], start.time = g_sel - 1)
  quantile_delay <- quantile(sfit_delay, 0.5) # median
  res_delay.dt <- data.table(data.frame(quantile_delay), keep.rownames = T)
  setnames(res_delay.dt, names(res_delay.dt), c("scenario", "q50", "low", "up"))
  res_delay.dt[, q50_s := q50 - g_sel]
  res_delay.dt[, low_s := low - g_sel]
  res_delay.dt[, up_s := up - g_sel]
  res_delay.dt[, med_fmt := paste0(comma(q50_s, accuracy = acc), " [", comma(low_s, accuracy = acc), ", ", comma(up_s, accuracy = acc), "]")]
  res_delay.dt[, c("alg", "g", "p1p2") := tstrsplit(scenario, ",", fixed=T)]
  res_delay.dt[, c("alg", "g", "p1p2") := list(str_trim(gsub(".*=","",alg)), str_trim(gsub(".*=","",g)), str_trim(gsub(".*=","",p1p2)))]
  return(res_delay.dt[, .(p1p2, alg, g, q50_s, med_fmt)])
}

### RESULTS
# Read in all scenarios and combine
atd_ind.dt <- data.table()
for (g in gammas) {
  for (p1 in p1s) {
    for (p2 in p2s) {
      if (!(p1 == 0.02 && p2 == 0.02)) { # did not do this combo
        for (a in alarms) {
          tmp.dt <- read_scenario_individual(g, p1, p2, a)
          atd_ind.dt <- rbindlist(list(atd_ind.dt, tmp.dt))
        }
      }
    }
  }
}

# Fit survival curves
sfit <-  survfit(Surv(t, status) ~ alg + g + p1p2 + alarm, data = atd_ind.dt)

# Publication plot of survival curves
sfit.dt <- data.table(surv_summary(sfit, data = atd_ind.dt))[, .(p1p2, alg, g, alarm, time, surv, upper, lower)]
# add first points of 1.0
sfit_add.dt <- data.table(CJ(p1p2 = unique(sfit.dt$p1p2), alg = unique(sfit.dt$alg), g = unique(sfit.dt$g), time = 0, surv = 1.0, upper = 1.0, lower = 1.0))
sfit.dt <- rbindlist(list(sfit.dt, sfit_add.dt))[order(p1p2, alg, g, time)]
sfit.dt[, c("alarm", "alarm_lower", "alarm_upper") := list(1 - surv, 1 - upper, 1 - lower)]


#sfit.dt[, p1p2 := factor(p1p2, levels = p1p2_levels, labels = c(TeX("$\alpha$"), "2", "3"))]
levels(sfit.dt$g) <- c(TeX("$\\Gamma_1=1$"), TeX("$\\Gamma_1=50$"))
levels(sfit.dt$p1p2) <- c(TeX("$p_l^0 = 0.01, p_2^0 = 0.01$"), TeX("$p_l^0 = 0.01, p_2^0 = 0.02$"), TeX("$p_l^0 = 0.02, p_2^0 = 0.01$"))

vlines.dt <- data.table(g = levels(sfit.dt$g), vline = c(1, 50))

ggplot(sfit.dt, aes(x = time, y = alarm, ymin=alarm_lower, ymax=alarm_upper, fill=alg, color=alg)) +
  facet_grid(p1p2 ~ g, labeller = label_parsed) + xlim(0, 150) +
  geom_step() + geom_stepribbon(alpha=0.5, color=NA, show.legend=F) + # to do: add CI
  geom_vline(aes(xintercept = vline), data=vlines.dt, color = "red") + 
  xlab("Time (weeks)") + ylab("Cumulative Probability of Alarm in Location 1") + 
  scale_color_discrete(name = "Algorithm") + 
  theme_bw() +
  theme(legend.position = "bottom")
  

#ggsave(paste(output_path, "survival_curves.pdf", sep="/"), width=12, height=6)
ggsave(paste(output_path, "survival_curves.pdf", sep="/"), width=6.5, height=6.5)

# Publication Table of Results
fap.dt <- false_alarm_prob(atd_ind.dt)
med1.dt <- median_cond_delay(atd_ind.dt, 1)
med50.dt <- median_cond_delay(atd_ind.dt, 50)
med.dt <- rbindlist(list(med1.dt, med50.dt))

tor.dt <- merge(med.dt, fap.dt[, .(p1p2, alg, g, p, fprob_fmt)], by = c("p1p2", "alg", "g"))
tor.dt <- tor.dt[, .(p1p2, alg, g, fprob_fmt, med_fmt)]
tor.dt <-  dcast(tor.dt, p1p2 + alg ~ g, value.var = c("fprob_fmt", "med_fmt"))
tor.dt

fwrite(tor.dt, paste(output_path, "table_of_results_raw.csv", sep="/"))

