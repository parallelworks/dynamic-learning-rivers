#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/outgoing-tomcat"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/outgoing-tomcat"
export PW_PARENT_NAME="inline.outgoing-tomcat"
export PW_WORKFLOW_NAME="inline.outgoing-tomcat"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.outgoing-tomcat-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWUyMmFjNGUyNjRmZTUxODM0YjRkIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlZTIyYWM0ZTI2NGZlNTE4MzRiNGQiXSwiZXhwIjoxNzg0MDQ5NDUwLCJpYXQiOjE3ODE0NTc0NTAsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.nAQIIFBBkC0W53THq-XiiYEf3ovDJlJrw8CuJ8h1hmwz7EjfHKrY7yXk7xb5WlwsPxnYvCK_lWb14yL-Q_Tuw7yRWfxZ5F1jQ4vFuWGuY4xq54uZvL2VN5bz9nbq689XCgEJRPYlbgyera9ojyaOa3kmgSW-No_I-0JFK3IvfUWfk-3lkoGRovN8JuGFRWU0yWsTQLab-HlL-2NInAB-5soHVDO7glIBadAZ-hq_zJXcKg7AKA82IHC9PRVV7YKRUPZqk6aoUdJBIvAm82swLg0RMYW5pYDpKfyZsxmItvNVbxzfG91jf-5HYFHZRrjzUy38hE_j0QTsrVudK95fl1_J_ifvtpAzOquwsO2udkZFXEUF4jXG6AQx7d8ZvzlK3RrJp7arfaKEr7wbhgOSEaRxa1BqXsoBH9VmcL1hChrIbA22jz-tkgwzXamxaiptoEBnBvkyVrkDgVMiECn4te3Uq8mPdzp9JkQvhuLTEo4dcADBwM1Cr5rHr6M_Esvqy7F5GG_O2EqGZW-HTtZ5BOtGQ-EmelRHT9ylBNEMw32n62Bky6kuOHmwqXg2een-CXbOK4l4HDz728k5YwKxC7mSdeh9z7mZoSMJcbkz_S-PDt1bhttVlBONi8OD5tYGUHMkdAJ9ciYbuJb8Ph2RUFomfns_RM0awd2RJ5oeeM0"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="outgoing-tomcat"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/outgoing-tomcat/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/outgoing-tomcat/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/outgoing-tomcat

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
  cat << EOF > sl.Apr-2023-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Apr-2023-log10-gss.${ii}
#SBATCH --output=sl.std.out.Apr-2023-log10-gss.${ii}
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
  sbatch sl.Apr-2023-log10-gss.${ii}.sbatch
done
echo "===================================="

