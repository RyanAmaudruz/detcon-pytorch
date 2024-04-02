#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --output=odin.out
#SBATCH --job-name=odin
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate obdet

srun python pretrain.py
