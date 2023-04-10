#!/bin/bash
#=========================
# Postprocessing orchestration
#=========================

# ASSUMING THAT THIS SCRIPT IS RUNNING IN ./dynamic-learning-rivers/scripts

miniconda_loc=$1
my_env=$2

# Activate Conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate $my_env

# Step 1: Use Pandas to flatten all results from each ML model
python post_01_flatten.py

# Step 2: Cut data to CONUS, Remove sites already sampled, sort
post_02_filter_sort.sh

