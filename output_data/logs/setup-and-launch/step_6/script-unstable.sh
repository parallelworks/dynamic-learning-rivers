#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/frank-polecat"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/frank-polecat"
export PW_PARENT_NAME="inline.frank-polecat"
export PW_WORKFLOW_NAME="inline.frank-polecat"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.frank-polecat-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWU4NmIyMjE1NzI3MTkzMjY1NzQ2Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlZTg2YjIyMTU3MjcxOTMyNjU3NDYiXSwiZXhwIjoxNzg0MDUxMDUxLCJpYXQiOjE3ODE0NTkwNTEsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.BzuLMNRFzYQvaRN-DDcmbYFzXPa53yK0zxCM7ujViRWmqbSkwsMGMaRP6kigR08ahXtXA0uXPlK8LJnRoYCUywvNM-rVVodkx7NFcaYxrA_VJ7YAWa5ZTEbMOFY_dI-_WhQmgPJmbW6o8gKtwM8xuDqBmlN8Y_PIUTlDTRphnhYiMS77Zxi4K4pEVgWdjW_EwMzft8SbAM97pqGDZTshcc4u31s0LVvTJvyqiwAb6UoNky_bbmPAeSU234F3WMTdViChR2wvXJENcxbdqzK0tLslw7c2kChrjb9rRoWZicHGC8TWssrvDL3n4oLg2IPEBbArZJ94A65zQqJOc9TgPbOk3AXFsgkKrZ7OIsoyTOaZtoLZUHzKHpOob_T0pEripiwRaDkNh3Rzgm801Ly95-F6jrFPu9cLOpqNHyUAXjYERcGFN4mK6s1ZEapkP-oFWWyEiIoeflt18DrXMlPJN2MZFeLsCfBOAs-lHAS-OXwlPiv-pILXjvFPypL8CJgQ7uyIo-cgawu9DQJ22-540OxGNv5jGMnwoagQSKVnLCnFegyZEjRq6Jtx2xlozibUjtKMd2uxmAHIZwnWFeVU305bOE2Qu2YW21Hjha6na0LJagRGPKTrnpBE2hK9nQRWpbb4Q98hZJTefQlYPNKm9yNHa1z_RSuDgTZ8qLHIXPw"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="frank-polecat"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/frank-polecat/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/frank-polecat/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/frank-polecat

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
  cat << EOF > sl.May-2023-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=May-2023-log10-gss.${ii}
#SBATCH --output=sl.std.out.May-2023-log10-gss.${ii}
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
  sbatch sl.May-2023-log10-gss.${ii}.sbatch
done
echo "===================================="

