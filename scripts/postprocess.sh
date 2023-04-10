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

# Send a copy of the key output to ../output_data
#cp post_01_output_ml_predict_avg.csv ../output_data/unfiltered_predict_output_avg.csv
#cp post_01_output_ml_predict_std.csv ../output_data/unfiltered_predict_output_std.csv
cp post_02_output_ml_pred_avg_filtered.csv ../output_data/filtered_predict_output.csv
