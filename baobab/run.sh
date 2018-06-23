#!/bin/env bash

#SBATCH -J DRAMA
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
##SBATCH --constraint=V6
#SBATCH -p dpt
#SBATCH --output=slurm-test-%J.out
#SBATCH -t 12:00:00
#SBATCH --mem=4000
##SBATCH --mail-type=FAIL

tfcpu $1 $2 $3 $4 $5 $6 $7

