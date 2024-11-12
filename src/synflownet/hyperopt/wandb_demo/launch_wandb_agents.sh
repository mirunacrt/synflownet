#!/bin/bash

# Purpose: Script to allocate a node and run a wandb sweep agent on it
# Usage: sbatch launch_wandb_agent.sh <SWEEP_ID>

#SBATCH --job-name=wandb_sweep_agent
#SBATCH --array=1-54
#SBATCH --time=23:59:00
#SBATCH --output=slurm_output_files/%x_%N_%A_%a.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition def

source activate sfn
echo "Using environment={$PYENV_VERSION}"

# launch wandb agent
wandb agent --count 1 --entity valencelabs --project gflownet $1
