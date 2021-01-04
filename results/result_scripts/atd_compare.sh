#!/bin/bash
#SBATCH --job-name=surveillance_atd_50_2_1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfairley@stanford.edu
ml julia

julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.01 0.01 1000 1
julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.01 0.02 30000 1
julia results/result_scripts/atd_compare.jl 0.1 0.1 1 0.02 0.01 1000 1
julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.01 0.01 1000 1
julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.01 0.02 30000 1
julia results/result_scripts/atd_compare.jl 0.1 0.1 50 0.02 0.01 1000 1

julia results/result_scripts/atd_compare.jl 0.05 0.05 1 0.01 0.01 1000 0
julia results/result_scripts/atd_compare.jl 0.05 0.05 1 0.01 0.02 30000 0
julia results/result_scripts/atd_compare.jl 0.05 0.05 1 0.02 0.01 1000 0
julia results/result_scripts/atd_compare.jl 0.05 0.05 50 0.01 0.01 1000 0
julia results/result_scripts/atd_compare.jl 0.05 0.05 50 0.01 0.02 30000 0
julia results/result_scripts/atd_compare.jl 0.05 0.05 50 0.02 0.01 1000 0