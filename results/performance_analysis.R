library(here)
library(data.table)
library(survival)
library(survminer)
results_path <- here("results", "tmp", "mfairley")
header <- c("p1", "p2", "a0", "a1", "a2") # assuming 2 locations

# Scenario: X
read_scenario_alg <- function(alg) {
  atd_alg.dt <- fread(paste(results_path, paste0("atd_", alg, "_1.csv"), sep="/"), col.names = header)
  atd_alg.dt[, t := 1:.N]
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

read_scenario <- function() {
  atd_constant.dt <- individual_format(read_scenario_alg("constant"))
  atd_random.dt <-individual_format(read_scenario_alg("random"))
  atd_thompson.dt <- individual_format(read_scenario_alg("thompson"))
  atd_evsi.dt <- individual_format(read_scenario_alg("evsi"))

  atd.dt <- rbindlist(list(constant=atd_constant.dt,random=atd_random.dt,thompson=atd_thompson.dt,evsi=atd_evsi.dt), use.names = T, idcol = "alg")
  return(atd.dt)
}

survival_analysis <- function (atd_ind.dt) {
  s <- survfit(Surv(t, status) ~ alg, data = atd_ind.dt)
}

atd_ind.dt <- read_scenario()
s <- survival_analysis(atd_ind.dt)
ggsurvplot(s, conf.int = T)


