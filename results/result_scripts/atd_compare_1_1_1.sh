#!/bin/bash
#SBATCH --job-name=surveillance_atd_1_1_1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfairley@stanford.edu
ml julia
julia --machine-file <(srun hostname -s) -t $SLURM_CPUS_PER_TASK results/result_scripts/atd_compare_1_1_1.jl
