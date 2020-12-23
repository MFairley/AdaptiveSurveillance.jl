#!/bin/bash
#SBATCH --job-name=surveillance_atd
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfairley@stanford.edu
ml julia
# export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
julia --machine-file <(srun hostname -s) -t $SLURM_CPUS_PER_TASK results/atd_compare.jl