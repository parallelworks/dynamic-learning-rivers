#!/bin/bash
#=========================
# Postprocessing orchestration
#=========================

miniconda_loc=$1
my_env=$2

# Activate Conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate $my_env

# Use Pandas to flatten all results from each ML model
python post_01_flatten.py

# Filter sites that have already been sampled
post_02_filter.sh

