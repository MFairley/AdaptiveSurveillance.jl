#!/bin/bash
#SBATCH --job-name=surveillance
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfairley@stanford.edu
#SBATCH --array=13-19
ml julia

case $SLURM_ARRAY_TASK_ID in
    # isotonic 0.1 0.1
    1) julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.01 0.01 I 1000 1 ;;
    2) julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.01 0.02 I 30000 1 ;;
    3) julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.02 0.01 I 1000 1 ;;
    4) julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.01 0.01 I 1000 1 ;;
    5) julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.01 0.02 I 30000 1 ;;
    6) julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.02 0.01 I 1000 1 ;;
    # isotonic 0.05 0.05
    7) julia results/result_scripts/atd_compare.jl 0.05 0.05 1 0.01 0.01 I 1000 0 ;;
    8) julia results/result_scripts/atd_compare.jl 0.05 0.05 1 0.01 0.02 I 30000 0 ;;
    9) julia results/result_scripts/atd_compare.jl 0.05 0.05 1 0.02 0.01 I 1000 0 ;;
    10) julia results/result_scripts/atd_compare.jl 0.05 0.05 50 0.01 0.01 I 1000 0 ;;
    11) julia results/result_scripts/atd_compare.jl 0.05 0.05 50 0.01 0.02 I 30000 0 ;;
    12) julia results/result_scripts/atd_compare.jl 0.05 0.05 50 0.02 0.01 I 1000 0 ;;
    # Logistic 0.1 0.1
    13) julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.01 0.01 L 2 1 ;;
    14) julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.01 0.02 L 2 1 ;;
    16) julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.02 0.01 L 2 1 ;;
    17) julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.01 0.01 L 2 1 ;;
    18) julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.01 0.02 L 2 1 ;;
    19) julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.02 0.01 L 2 1 ;;
esac