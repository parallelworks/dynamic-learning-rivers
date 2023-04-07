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

#=============================================
# Intake training and predict data
#=============================================
python prep_01_intake_train.py
python prep_02_intake_predict.py

#=============================================
# Colocate (in parallel) using data/tools in global-river-databases
#=============================================

#-------------Training Data----------------
# Run colocate in header mode first to get the header
# and ensure that the data are prepared in advance.
prep_dir=$(pwd)
cd ${grd_abs_path}/scripts
./run_colocate.sh header 42.0 42.0
cd $prep_dir

# Run multiple instances of run_colocate, one for each site

# For now, do not run as sbatch - too many docker pulls!
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
# Paste
#=============================================

#-------------Training Data----------------
