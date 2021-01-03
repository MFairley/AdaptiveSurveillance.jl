library(here)
library(data.table)
library(survival)
library(survminer)
library(ggplot2)
library(scales)
library(stringr)
library(latex2exp)
results_path <- here("results", "tmp", "mfairley")
header <- c("p1", "p2", "p3", "p4", "p5","a0", "a1", "a2", "a3", "a4", "a5") # assuming 5 locations

# Scenario: g, p1, p2
read_scenario_alg <- function(alg, g, p1, p2) {
  atd_alg.dt <- fread(paste(results_path, paste0("atd_", alg, "_", g, "_", p1, "_", p2, ".csv"), sep="/"), col.names = header)
  atd_alg.dt[, t := 1:.N]
  atd_alg.dt[, alg := alg]
  atd_alg.dt[, g := factor(g)]
  atd_alg.dt[, p1p2 := factor(paste0(p1, "_", p2))]
  return(atd_alg.dt)
}

read_scenario <- function(g, p1, p2) {
  atd_constant.dt <- read_scenario_alg("constant", g, p1, p2)
  atd_random.dt <- read_scenario_alg("random", g, p1, p2)
  atd_thompson.dt <- read_scenario_alg("thompson", g, p1, p2)
  atd_evsi.dt <- read_scenario_alg("evsi", g, p1, p2)
  atd.dt <- rbindlist(list(constant=atd_constant.dt,random=atd_random.dt,thompson=atd_thompson.dt,evsi=atd_evsi.dt), use.names = T, idcol = "alg")
  return(atd.dt)
}

individual_format <- function (atd.dt) { # assuming outbreak in location 1 to do: make this generalize to L locations
  d1 <- data.table(t = rep(atd.dt$t, atd.dt$a1), status = 1)[!is.na(t)] # <- outbreak location
  d2 <- data.table(t = rep(atd.dt$t, atd.dt$a0), status = 0)[!is.na(t)] # to do: make this not throw a warning
  d3 <- data.table(t = rep(atd.dt$t, atd.dt$a2), status = 0)[!is.na(t)]
  d4 <- data.table(t = rep(atd.dt$t, atd.dt$a3), status = 0)[!is.na(t)]
  d5 <- data.table(t = rep(atd.dt$t, atd.dt$a4), status = 0)[!is.na(t)]
  d6 <- data.table(t = rep(atd.dt$t, atd.dt$a5), status = 0)[!is.na(t)]
  ind.dt <- rbindlist(list(d1, d2, d3, d4, d5, d6))
  return(ind.dt)
}

read_scenario_individual <- function(g, p1, p2) {
  atd_constant.dt <- individual_format(read_scenario_alg("constant", g, p1, p2))
  atd_random.dt <-individual_format(read_scenario_alg("random", g, p1, p2))
  atd_thompson.dt <- individual_format(read_scenario_alg("thompson", g, p1, p2))
  atd_evsi.dt <- individual_format(read_scenario_alg("evsi", g, p1, p2))

  atd.dt <- rbindlist(list(constant=atd_constant.dt,random=atd_random.dt,thompson=atd_thompson.dt,evsi=atd_evsi.dt), use.names = T, idcol = "alg")
  atd.dt[, alg := factor(alg, levels = c("constant", "evsi", "thompson", "random"), labels = c("Clairvoyance", "Profile Likelihood", "Thompson Sampling", "Uniform Random"))]
  atd.dt[, g := factor(g,)]
  atd.dt[, p1p2 := paste0(p1, "_", p2)]
  return(atd.dt)
}

### RESULTS
# Read in all scenarios and combine
atd_ind_1_1_1.dt <- read_scenario_individual(1, 0.01, 0.01)
atd_ind_1_1_2.dt <- read_scenario_individual(1, 0.01, 0.02)
atd_ind_1_2_1.dt <- read_scenario_individual(1, 0.02, 0.01)
atd_ind_50_1_1.dt <- read_scenario_individual(50, 0.01, 0.01)
atd_ind_50_1_2.dt <- read_scenario_individual(50, 0.01, 0.02)
atd_ind_50_2_1.dt <- read_scenario_individual(50, 0.02, 0.01)
atd_ind.dt <- rbindlist(list(atd_ind_1_1_1.dt, atd_ind_1_1_2.dt,
                             atd_ind_1_2_1.dt, atd_ind_50_1_1.dt,
                             atd_ind_50_1_2.dt, atd_ind_50_2_1.dt))

# Fit survival curves
sfit <-  survfit(Surv(t, status) ~ alg + g + p1p2, data = atd_ind.dt)

# Publication plot
sfit.dt <- data.table(surv_summary(sfit))[, .(p1p2, alg, g, time, surv, upper, lower)]
# add first point
sfit_add.dt <- data.table(p1p2 = unique(sfit.dt$p1p2), alg = unique(sfit.dt$alg), g = unique(sfit.dt$g), time = 0, surv = 1.0, upper = 1.0, lower = 1.0)
sfit.dt <- rbindlist(list(sfit.dt, sfit_add.dt))[order(p1p2, alg, g, time)]
sfit.dt[, c("alarm", "alarm_lower", "alarm_upper") := list(1 - surv, 1 - upper, 1 - lower)]

ggplot(sfit.dt, aes(x = time, y = alarm, ymin=alarm_lower, ymax=alarm_upper, fill=alg, color=alg)) +
  facet_grid(rows = vars(p1p2), cols = vars(g)) +
  geom_step() + # to do: add CI +
  xlab("Time (weeks)") + ylab("Cumulative Probability of Alarm in Location 1") + 
  scale_color_discrete(name = "Algorithm") + 
  theme(legend.position = "bottom")

ggsave("survival_curves.pdf", width=8, height=8)

# False Alarm Probability for location 1 only
false_alarm_prob <- function(g_sel, acc=0.01) {
  sfit_false <- survfit(Surv(t, status) ~ alg + g + p1p2, data = atd_ind.dt[g == g_sel])  
  res_false <- summary(sfit_false, times = g_sel)
  res_false.dt <- data.table(scenario = rownames(res_false$table), fprob = 1 - res_false$surv, fprob_low = 1 - res_false$upper, fprob_high = 1 - res_false$lower)
  res_false.dt[, fprob_fmt := paste0(comma(fprob, accuracy = acc), " [", comma(fprob_low, accuracy = acc), ", ", comma(fprob_high, accuracy = acc), "]")]
  return(res_false.dt[, .(scenario, fprob_fmt)])
}

fap1.dt <- false_alarm_prob(1)
fap50.dt <- false_alarm_prob(50)
fap.dt <- rbindlist(list(fap1.dt, fap50.dt))

# False Alarm Probability for all locations


# Median Delay
median_cond_delay <- function(g_sel, acc=1.0) {
  sfit_delay <-  survfit(Surv(t, status) ~ alg + g + p1p2, data = atd_ind.dt[g == g_sel], start.time = g_sel - 1)
  quantile_delay <- quantile(sfit_delay, 0.5) # median
  res_delay.dt <- data.table(data.frame(quantile_delay), keep.rownames = T)
  setnames(res_delay.dt, names(res_delay.dt), c("scenario", "q50", "low", "up"))
  res_delay.dt[, q50_s := q50 - g_sel]
  res_delay.dt[, low_s := low - g_sel]
  res_delay.dt[, up_s := up - g_sel]
  res_delay.dt[, med_fmt := paste0(comma(q50_s, accuracy = acc), " [", comma(low_s, accuracy = acc), ", ", comma(up_s, accuracy = acc), "]")]
  return(res_delay.dt[, .(scenario, med_fmt)])
}

med1.dt <- median_cond_delay(1)
med50.dt <- median_cond_delay(50)
med.dt <- rbindlist(list(med1.dt, med50.dt))


# Publication Table of Results
tor.dt <- merge(fap.dt, med.dt, by = "scenario")
tor.dt[, c("alg", "g", "p1p2") := tstrsplit(scenario, ",", fixed=T)]
tor.dt[, c("alg", "g", "p1p2") := list(str_trim(gsub(".*=","",alg)), str_trim(gsub(".*=","",g)), str_trim(gsub(".*=","",p1p2)))]
tor.dt <- tor.dt[, .(p1p2, alg, g, fprob_fmt, med_fmt)]
# <- 
tor.dt <-  dcast(tor.dt, p1p2 + alg ~ g, value.var = c("fprob_fmt", "med_fmt"))
#tor.dt <- tor.dt[, .(p1p2, alg, fprob_fmt_1, med_fmt_1, fprob_fmt_50, med_fmt_50)]
fwrite(tor.dt, paste(results_path, "table_of_results_raw.csv", sep="/"))

tor.dt







