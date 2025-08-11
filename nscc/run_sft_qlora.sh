#!/bin/bash
#PBS -q normal
#PBS -N sft
#PBS -l select=1:ngpus=2
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -o /home/users/nus/e1503348/scratch/ay2425_sem2/cs6207/out/

# Load necessary modules
module load cuda/12.2.2 cudnn/12-8.9.4.25 gcc/11.2.0

# Activate conda environment
export PATH=/home/users/nus/e1503348/miniconda3/bin:$PATH
source activate llamafactory

# Define paths
BASE_DIR=/home/users/nus/e1503348/scratch
PROJECT_DIR=${BASE_DIR}/dissertation/LLaMA-Factory
LOG_FILE=${BASE_DIR}/dissertation/out/sft_qlora_${PBS_JOBID}.out

export HF_TOKEN=$(<${BASE_DIR}/hf_token.txt)

# CD to working directory
cd ${PROJECT_DIR}

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_qlora/llama_primus_lora_sft_otfq.yaml \
>> ${LOG_FILE} 2>&1


