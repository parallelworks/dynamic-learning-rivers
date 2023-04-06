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

# Intake training and predict data
python prep_01_intake_train.py
python prep_02_intake_predict.py

# Colocate (in parallel) using data/tools in global-river-databases
#awk -F, -v grd=grd_abs_path ' NR>1 {system("ls -alh; echo "$2" "$3" "$4)}' prep_01_output.csv

# Run colocate in header mode first to get the header
# and ensure that the data are prepared in advance.
prep_dir=$(pwd)
cd ${grd_abs_path}/scripts
./run_colocate.sh header 42.0 42.0
cd $prep_dir

# Run multiple instances of run_colocate, one for each site

# For now, do not run as sbatch - too many docker pulls!
#awk -F, -v grd=${grd_abs_path} ' NR>1 {system("cd "grd"/scripts; sbatch -n 1 -c 1 --mem=0 --output=$HOME/slurm-%j.out --wrap \"./run_colocate.sh "$1" "$2" "$3"\"")}' prep_01_output.csv

# Run on head node to minimize docker pulls
awk -F, -v grd=${grd_abs_path} ' NR>1 {system("cd "grd"/scripts; ./run_colocate.sh "$1" "$2" "$3" &")}' prep_01_output.csv

# Concatenate

# Paste
