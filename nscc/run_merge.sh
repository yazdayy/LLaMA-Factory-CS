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
unset CONDA_PREFIX
unset _CONDA_EXE
unset CONDA_SHLVL
unset CONDA_PYTHON_EXE
unset CONDA_DEFAULT_ENV
source activate llamafactory

# Define paths
BASE_DIR=/home/users/nus/e1503348/scratch
PROJECT_DIR=${BASE_DIR}/dissertation/LLaMA-Factory
LOG_FILE=${BASE_DIR}/dissertation/out/merge_${PBS_JOBID}.out
MODEL_CACHE_DIR=${BASE_DIR}/models

BASE_MODEL_PATH=${MODEL_CACHE_DIR}/backbone/models--trendmicro-ailab--Llama-Primus-Base/snapshots/5295f5a992fbbf06c0754e83864c44ddcc431335
ADAPTER_PATH=${MODEL_CACHE_DIR}/sft/llama-primus-instruct/filtered-lang/qlora_adapter
SAVE_PATH=${MODEL_CACHE_DIR}/sft/llama-primus-instruct/filtered-lang/qlora

export HF_TOKEN=$(<${BASE_DIR}/hf_token.txt)

# CD to working directory
cd ${PROJECT_DIR}

python model_merge.py \
--base_model_path ${BASE_MODEL_PATH} \
--adapter_path ${ADAPTER_PATH} \
--save_path ${SAVE_PATH} \
>> ${LOG_FILE} 2>&1


