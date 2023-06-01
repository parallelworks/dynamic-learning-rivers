#!/bin/bash
#======================
# Data intake/preprocessing
# orchestration script
#======================

# ASSUMING THAT THIS SCRIPT IS RUNNING IN ./dynamic-learning-rivers/scripts

miniconda_loc=$1
my_env=$2
grd_abs_path=$3

# Activate Conda
source ${miniconda_loc}/etc/profile.d/conda.sh
conda activate $my_env

# Delete any .csv files here to ensure all data are
# created from fresh
rm -f *.csv

# Set the name of the target feature
target_name="Normalized_Respiration_Rate_mg_DO_per_H_per_L_sediment"

#=============================================
# Steps 1 & 2: Intake training and predict data
#=============================================
python prep_01_intake_train.py --target_name $target_name
python prep_02_intake_predict.py

#=============================================
# Steps 3 & 4: Colocate using data/tools in global-river-databases
#=============================================

#-------------Training Data----------------
# Run colocate in header mode first to get the header
# and ensure that the data are prepared in advance.
prep_dir=$(pwd)
cd ${grd_abs_path}/scripts
./run_colocate.sh header 42.0 42.0
cd $prep_dir

# Run multiple instances of run_colocate, one for each site
# Careful - if running on many nodes, this can result in 
# many docker pulls!
awk -F, -v grd=${grd_abs_path} ' NR>1 {system("cd "grd"/scripts; sbatch -n 1 -c 1 --mem=0 --output=$HOME/slurm-%j.out --wrap \"./run_colocate.sh "$1" "$2" "$3"\"")}' prep_01_output_train.csv

# Wait for queue to empty
n_squeue="2"
squeue_wait=10
while [ $n_squeue -gt 1 ]
do
    # Wait first - sbatch launches may take  
    # a few seconds to register on squeue!
    echo "Monitor waiting "${squeue_wait}" seconds..." >> prep_03_monitor.log
    sleep $squeue_wait
    n_squeue=$(squeue | wc -l )
    echo "Found "${n_squeue}" lines in squeue." >> prep_03_monitor.log
done
echo "No more pending jobs in squeue. Proceed." >> prep_03_monitor.log

# Run on head node to minimize docker pulls (add " &" for background running).
#awk -F, -v grd=${grd_abs_path} ' NR>1 {system("cd "grd"/scripts; ./run_colocate.sh "$1" "$2" "$3)}' prep_01_output.csv

# Concatenate
# All files end up in grd_abs_path/scripts (where it works)
# concatenate them together
sudo cat ${grd_abs_path}/scripts/colocated.??????????.tmp >> ${grd_abs_path}/scripts/colocated.header.tmp

# Archive colocation step
cp ${grd_abs_path}/scripts/colocated.header.tmp prep_03_output_colocated_train.csv

# Clean up
sudo rm -f ${grd_abs_path}/scripts/*.tmp

echo -------------Predict Data----------------
# Run colocate in header mode first to get the header
# and ensure that the data are prepared in advance.
prep_dir=$(pwd)
cd ${grd_abs_path}/scripts
./run_colocate.sh header 42.0 42.0
cd $prep_dir

# Run multiple instances of run_colocate, one for each site

# For now, do not run as sbatch - too many docker pulls!
awk -F, -v grd=${grd_abs_path} ' NR>1 {system("cd "grd"/scripts; sbatch -n 1 -c 1 --mem=0 --output=$HOME/slurm-%j.out --wrap \"./run_colocate.sh "$1" "$2" "$3"\"")}' prep_02_output_predict.csv

# Wait for queue to empty
n_squeue="2"
squeue_wait=10
while [ $n_squeue -gt 1 ]
do
    # Wait first - sbatch launches may take 
    # a few seconds to register on squeue!
    echo "Monitor waiting "${squeue_wait}" seconds..." >> prep_04_monitor.log
    sleep $squeue_wait
    n_squeue=$(squeue | wc -l )
    echo "Found "${n_squeue}" lines in squeue." >> prep_04_monitor.log
done
echo "No more pending jobs in squeue. Proceed." >> prep_04_monitor.log

# Run on head node to minimize docker pulls
#awk -F, -v grd=${grd_abs_path} ' NR>1 {system("cd "grd"/scripts; ./run_colocate.sh "$1" "$2" "$3" &")}' prep_02_output.csv

# Concatenate
# All files end up in grd_abs_path/scripts (where it works)
# concatenate them together
sudo cat ${grd_abs_path}/scripts/colocated.??????????.tmp >> ${grd_abs_path}/scripts/colocated.header.tmp

# Archive colocation step
cp ${grd_abs_path}/scripts/colocated.header.tmp prep_04_output_colocated_predict.csv

# Clean up
sudo rm -f ${grd_abs_path}/scripts/*.tmp

#=============================================
# Step 5: Get larger predict data
#=============================================
large_predict_csv=${grd_abs_path}/scripts/step_10_output.csv

# Copy it here because I want an archive of this in
# dynamic-learning-rivers and I don't want to have to
# pass the location of the file into prep_06_merge.py.
cp $large_predict_csv prep_05_output_large_predict.csv

#=============================================
# Merge/paste/cut columns
#=============================================
python prep_06_merge.py --target_name $target_name

