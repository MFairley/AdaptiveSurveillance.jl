# Adaptive Surveillance for Endemic Disease Outbreaks

This repository is the official implementation of [Surveillance for Endemic Infectious Disease Outbreaks: Adaptive
Sampling Using Profile Likelihood Estimation]()

Outbreaks of an endemic infectious disease can occur when the disease is introduced
into a highly susceptible subpopulation or when the disease enters a network of connected individuals. For example, significant HIV outbreaks among people who inject drugs have occurred in at least half a dozen U.S. states in recent years. This motivates the current study: how can limited testing resources be allocated across geographic regions to rapidly detect outbreaks of an endemic infectious disease? 

We develop an adaptive sampling algorithm that uses profile likelihood to estimate the distribution of the number of positive tests that would occur for each location in a future time period if that location were sampled. Sampling is performed in the location with the highest estimated probability of triggering an outbreak alarm in the next time period. The alarm function is determined by a semiparametric likelihood ratio test. We compare the profile likelihood sampling (PLS) method numerically to uniform random sampling (URS) and Thompson sampling (TS). TS was worse than URS when the
outbreak occurred in a location with lower initial prevalence than other locations. PLS had lower time to outbreak detection than TS in some but not all scenarios, but was always better than URS even when the outbreak occurred in a location with a lower initial prevalence than other locations. PLS provides an effective and reliable method for rapidly detecting endemic disease outbreaks that is robust to this uncertainty

## Running the Model

For a fair comparison of the different methods, we set the ARL runtime when there is no alarm for the different methods to be the same. For our paper, we first ran our EVSI method with an alpha value of 2 which resulted in an ARL value of 56 for the three different scenarios. To do this, we ran the following code. 

```
# case 1
julia --project=. -t 4 results/result_scripts/gq_tests.jl 0.1 0.1 Lr 1 0.01 0.01 1000 300 1 0
# case 2
julia --project=. -t 4 results/result_scripts/gq_tests.jl 0.1 0.1 Lr 1 0.01 0.02 1000 300 1 0
# case 3
julia --project=. -t 4 results/result_scripts/gq_tests.jl 0.1 0.1 Lr 1 0.02 0.01 1000 300 1 0
```

We then calibrate the values of alpha for the other methods to have the same value ARL value of 56. This can be achieved by changing the target arl parameter in the atd_compare.jl script. 


