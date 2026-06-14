#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/ethical-perch"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/ethical-perch"
export PW_PARENT_NAME="inline.ethical-perch"
export PW_WORKFLOW_NAME="inline.ethical-perch"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.ethical-perch-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWIxYzAyMjE1NzI3MTkzMjYzYjJiIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYjFjMDIyMTU3MjcxOTMyNjNiMmIiXSwiZXhwIjoxNzg0MDM3MDU2LCJpYXQiOjE3ODE0NDUwNTYsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.gcFKxJJletW9lj7hKKszjirf_37hwvQ1Q4XJNGlsDDJBQQL4F3VYSbKp8OyjrvpvJT43m-9ZbCXc1NhXlmjzqm2tE67WXGxoJxxGgNvgtHBm13UJk8k_CDqY9WSsi56wvKygCYnGRqY55oc_g7tDPTqs-1n2bjEdxq8tzKO1aUk5XQK92Fw-vAYBCGtB9g7Pk37VOdYw1oW2y6wnzRpkOenA8ujSZiCSvjIuAwDsKPl6hKETl2pRQKc-aRLEIkDc6xItzR0Iguq5-JkgFdNB0iD0J2WaObM6SR-ELX7Cp9GzUREC-tjiJR5ipeBZmAJO5bG9wwHraqpWZHFgzCVZ8AhWyRguourlt1e8IKSosesQU8oqymM7ft5imJNM8LecOiFCRJYKzuYbtt1QniRyd1n7GztqqxHbmHyWpolcHcaC261i5qScW6i7MGc5nlA2PHU1s2ampAaeF74k-tOjjSMYS0WIQ9obB1YGktuMJcHfqJfRGZTylbL1rZf8ySCrOLwZ8-5jQMqCHYjS630UbroFzZf-kN9FPsaKOuD5wW3nqYkua0qtyCwrJADw0WFU9685TF0exh7Uf2OCtbnAtoh_wEe2p_1I_91MWwEQL4gj3dcmPZUusM4FLfpemjIhapU-MF1W2cVsFjrDJDpDZ0dDpbaURyuIOiIj4gRq8A0"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="ethical-perch"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/ethical-perch/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/ethical-perch/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/ethical-perch

echo "===================================="
echo "Setting target name..."
target_name="Normalized_Respiration_Rate_mg_DO_per_H_per_L_sediment"
echo "Set target name to ${target_name}"
echo "===================================="
echo "Converting FPI correlation cutoff from integer 1-100 to decimal 0-1..."
echo "Starting with cutoff of 50"
superlearner_fpi_corr_cutoff=`echo 50 | awk '{print $1/100}'`
echo "Converted to $superlearner_fpi_corr_cutoff"
echo "===================================="
for (( ii=0; ii<10; ii++ ))
do
  # Launch a single SuperLearner job
  work_dir=${HOME}/$(basename https://github.com/parallelworks/dynamic-learning-rivers )/ml_models/sl_${ii}
  echo "=======> Deleting any existing data in ${work_dir}"
  rm -rf ${work_dir}
  echo "=======> Creating work dir: ${work_dir}"
  mkdir -p ${work_dir}
  echo "======> Building job card ${ii}"
  cat << EOF > sl.Oct-2022-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Oct-2022-log10-gss.${ii}
#SBATCH --output=sl.std.out.Oct-2022-log10-gss.${ii}
cd ${HOME}/$(basename https://github.com/parallelworks/sl_core)
./train_predict_eval.sh \
  ${HOME}/$(basename https://github.com/parallelworks/dynamic-learning-rivers)/scripts/prep_06_output_final_train.csv \
  ${HOME}/$(basename https://github.com/parallelworks/dynamic-learning-rivers)/scripts/prep_06_output_final_train.ixy \
  25 \
  ${HOME}/$(basename https://github.com/parallelworks/sl_core)/sample_inputs/superlearner_conf.py \
  ${HOME}/$(basename https://github.com/parallelworks/dynamic-learning-rivers)/ml_models/sl_${ii} \
  /home/${USER}/.miniconda3 \
  superlearner \
  true \
  true \
  false \
  false \
  8 \
  loky \
  ${target_name} \
  ${HOME}/$(basename https://github.com/parallelworks/dynamic-learning-rivers)/scripts/prep_06_output_final_predict \
  $superlearner_fpi_corr_cutoff
EOF
  echo "======> Launching SuperLearner ${ii}"
  sbatch sl.Oct-2022-log10-gss.${ii}.sbatch
done
echo "===================================="

