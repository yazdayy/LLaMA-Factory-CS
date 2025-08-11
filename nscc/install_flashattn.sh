#!/bin/bash
#PBS -q normal
#PBS -N merge
#PBS -l select=4:ncpus=8:mem=64gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /home/users/nus/e1503348/scratch/ay2425_sem2/cs6207/out/

# Load necessary modules
module load cuda/12.2.2 cudnn/12-8.9.4.25 gcc/11.2.0

# Activate conda environment
export PATH=/home/users/nus/e1503348/miniconda3/bin:$PATH
source activate llamafactory

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
