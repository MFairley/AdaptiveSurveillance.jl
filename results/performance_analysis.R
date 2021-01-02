library(here)
library(data.table)
library(survival)
library(survminer)
results_path <- here("results", "tmp", "mfairley")
header <- c("p1", "p2", "a0", "a1", "a2") # assuming 2 locations

# Scenario: g, p1, p2
read_scenario_alg <- function(alg, g, p1, p2) {
  atd_alg.dt <- fread(paste(results_path, paste0("atd_", alg, "_", g, "_", p1, "_", p2, ".csv"), sep="/"), col.names = header)
  atd_alg.dt[, t := 1:.N]
  atd_alg.dt[, t := t - g] # adjust to center around outbreak start time
  atd_alg.dt[, alg := alg]
  return(atd_alg.dt)
}

individual_format <- function (atd.dt) { # assuming for location 1
  d1 <- data.table(t = rep(atd.dt$t, atd.dt$a1), status = 1)[!is.na(t)]
  d2 <- data.table(t = rep(atd.dt$t, atd.dt$a2), status = 0)[!is.na(t)]
  d3 <- data.table(t = rep(atd.dt$t, atd.dt$a0), status = 0)[!is.na(t)]
  ind.dt <- rbindlist(list(d1, d2, d3))
  return(ind.dt)
}

read_scenario <- function(g, p1, p2) {
  atd_constant.dt <- read_scenario_alg("constant", g, p1, p2)
  atd_random.dt <- read_scenario_alg("random", g, p1, p2)
  atd_thompson.dt <- read_scenario_alg("thompson", g, p1, p2)
  atd_evsi.dt <- read_scenario_alg("evsi", g, p1, p2)
  
  atd.dt <- rbindlist(list(constant=atd_constant.dt,random=atd_random.dt,thompson=atd_thompson.dt,evsi=atd_evsi.dt), use.names = T, idcol = "alg")
  return(atd.dt)
}

read_scenario_individual <- function(g, p1, p2) {
  atd_constant.dt <- individual_format(read_scenario_alg("constant", g, p1, p2))
  atd_random.dt <-individual_format(read_scenario_alg("random", g, p1, p2))
  atd_thompson.dt <- individual_format(read_scenario_alg("thompson", g, p1, p2))
  atd_evsi.dt <- individual_format(read_scenario_alg("evsi", g, p1, p2))

  atd.dt <- rbindlist(list(constant=atd_constant.dt,random=atd_random.dt,thompson=atd_thompson.dt,evsi=atd_evsi.dt), use.names = T, idcol = "alg")
  return(atd.dt)
}

prob_false_alarm <- function (atd.dt, g) {
  n <- sum(atd.dt[, a1])
  e <- sum(atd.dt[t < 0, a1])
  p <- e / n
  hw <- sqrt(p * (1 - p) / n)
  return(c(p-hw, p+hw))
}

# 1, 0.01, 0.01
prob_false_alarm(read_scenario(1, 0.01, 0.01), 1)
atd_ind_1_1_1.dt <- read_scenario_individual(1, 0.01, 0.01)
atd_ind_1_1_1_filt.dt <- atd_ind_1_1_1.dt[t >= 0]
s_1_1_1 <- survfit(Surv(t, status) ~ alg, data = atd_ind_1_1_1_filt.dt)
s_1_1_1
ggsurvplot(s_1_1_1, conf.int = T) # this breaks unless have access to the data

# 1, 0.01, 0.02
prob_false_alarm(read_scenario(1, 0.01, 0.02), 1)
atd_ind_1_1_2.dt <- read_scenario_individual(1, 0.01, 0.02)
atd_ind_1_1_2_filt.dt <- atd_ind_1_1_2.dt[t >= 0]
s_1_1_2 <- survfit(Surv(t, status) ~ alg, data = atd_ind_1_1_2_filt.dt)
s_1_1_2

# 50, 0.01, 0.01
prob_false_alarm(read_scenario(50, 0.01, 0.01), 50)
atd_ind_50_1_1.dt <- read_scenario_individual(50, 0.01, 0.01)
atd_ind_50_1_1_filt.dt <- atd_ind_50_1_1.dt[t >= 0]
s_50_1_1 <- survfit(Surv(t, status) ~ alg, data = atd_ind_50_1_1_filt.dt)
s_50_1_1

# 50, 0.01, 0.02
prob_false_alarm(read_scenario(50, 0.01, 0.02), 50)
atd_ind_50_1_2.dt <- read_scenario_individual(50, 0.01, 0.02)
atd_ind_50_1_2_filt.dt <- atd_ind_50_1_2.dt[t >= 0]
s_50_1_2 <- survfit(Surv(t, status) ~ alg, data = atd_ind_50_1_2_filt.dt)
s_50_1_2

# 50, 0.02, 0.01
prob_false_alarm(read_scenario(50, 0.02, 0.01), 50)
atd_ind_50_2_1.dt <- read_scenario_individual(50, 0.02, 0.01)
atd_ind_50_2_1_filt.dt <- atd_ind_50_2_1.dt[t >= 0]
s_50_2_1 <- survfit(Surv(t, status) ~ alg, data = atd_ind_50_2_1_filt.dt)
s_50_2_1

