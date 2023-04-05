#!/bin/bash
#======================
# Data intake/preprocessing
# orchestration script
#======================

miniconda_loc=$1
my_env=$2
grd_abs_path=$3

# Activate Conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate $my_env

# Intake
python prep_01_intake.py

# Colocate (in parallel) using data/tools in global-river-databases
awk -F, -v grd=grd_abs_path ' NR>1 {system("ls -alh; echo "$2" "$3" "$4)}' prep_01_output.csv

# Concatenate

# Paste
