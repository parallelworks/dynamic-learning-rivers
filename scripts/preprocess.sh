#!/bin/bash
#======================
# Data intake/preprocessing
# orchestration script
#======================

miniconda_loc=$1
my_env=$2

# Activate Conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate $my_env

# Intake
python prep_01_intake.py

# Colocate (in parallel)
#awk '{system()}' prep_01_outpt.csv 

# Concatenate

# Paste
