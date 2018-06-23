#!/bin/env bash

#SBATCH -J SGAN
#SBATCH --partition=dpt-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=30000

#module load CUDA

# if you need to know the allocated CUDA device, you can obtain it here:
echo $CUDA_VISIBLE_DEVICES

srun tfpython $1 $2 $3 $4 $5 $6 $7

