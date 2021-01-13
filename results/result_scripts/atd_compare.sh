#!/bin/bash
#SBATCH --job-name=surveillance
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfairley@stanford.edu
#SBATCH --array=1-18
ml julia
# ARRAY_TASKS for mail-type if want email for every task in array
case $SLURM_ARRAY_TASK_ID in
    # Isotonic 0.1 0.1
    1) julia results/result_scripts/atd_compare.jl 0.1 0.1 I 1 0.01 0.01 1000 300 1 1 ;;
    2) julia results/result_scripts/atd_compare.jl 0.1 0.1 I 1 0.01 0.02 1000 300 1 1 ;;
    3) julia results/result_scripts/atd_compare.jl 0.1 0.1 I 1 0.02 0.01 1000 300 1 1 ;;
    4) julia results/result_scripts/atd_compare.jl 0.1 0.1 I 50 0.01 0.01 1000 300 1 1 ;;
    5) julia results/result_scripts/atd_compare.jl 0.1 0.1 I 50 0.01 0.02 1000 300 1 1 ;; # completing
    6) julia results/result_scripts/atd_compare.jl 0.1 0.1 I 50 0.02 0.01 1000 300 1 1 ;; # taking a long time
    # Isotonic 0.05 0.05
    7) julia results/result_scripts/atd_compare.jl 0.05 0.05 I 1 0.01 0.01 1000 300 0 1 ;;
    8) julia results/result_scripts/atd_compare.jl 0.05 0.05 I 1 0.01 0.02 1000 300 0 1 ;;
    9) julia results/result_scripts/atd_compare.jl 0.05 0.05 I 1 0.02 0.01 1000 300 0 1 ;;
    10) julia results/result_scripts/atd_compare.jl 0.05 0.05 I 50 0.01 0.01 1000 300 0 1 ;; # taking a long time
    11) julia results/result_scripts/atd_compare.jl 0.05 0.05 I 50 0.01 0.02 1000 300 0 1 ;; # taking a long time
    12) julia results/result_scripts/atd_compare.jl 0.05 0.05 I 50 0.02 0.01 1000 300 0 1 ;; # taking a long time
    # Logistic 0.1 0.1
    13) julia results/result_scripts/atd_compare.jl 0.1 0.1 L 1 0.01 0.01 1000 300 1 1 ;;
    14) julia results/result_scripts/atd_compare.jl 0.1 0.1 L 1 0.01 0.02 1000 300 1 1 ;; # taking a long time
    15) julia results/result_scripts/atd_compare.jl 0.1 0.1 L 1 0.02 0.01 1000 300 1 1 ;;
    16) julia results/result_scripts/atd_compare.jl 0.1 0.1 L 50 0.01 0.01 1000 300 1 1 ;;
    17) julia results/result_scripts/atd_compare.jl 0.1 0.1 L 50 0.01 0.02 1000 300 0 1 ;; # thompson taking a long time
    18) julia results/result_scripts/atd_compare.jl 0.1 0.1 L 50 0.01 0.02 1000 300 1 0 ;;
    19) julia results/result_scripts/atd_compare.jl 0.1 0.1 L 50 0.02 0.01 1000 300 1 1;; # taking a long time
esac