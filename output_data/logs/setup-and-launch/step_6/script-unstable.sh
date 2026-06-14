#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/tender-lamb"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/tender-lamb"
export PW_PARENT_NAME="inline.tender-lamb"
export PW_WORKFLOW_NAME="inline.tender-lamb"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.tender-lamb-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZTI1NjhjODk5MzVlNGEwYWM5ZGQ1Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlMjU2OGM4OTkzNWU0YTBhYzlkZDUiXSwiZXhwIjoxNzg0MDAxMTI4LCJpYXQiOjE3ODE0MDkxMjgsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.Mf4I3knfYrFblNarXs0UivCYXFI_Bchuog5cVBsqnxBZ1J4ZZ9lp3b98eH5Z85cer53JlxE4ACtJgxmbYqpkbvIpOvbwCIyBzemGYgN0GNCRPzN46er2k6sX1tSpxt15Bob90Men4Iwd2Pv_EVeupD_ZdR41mW_1i-Q8Pm_Yu2Zcxnq6K7N0oy6hcpkjAyTeIVJpyIu45BbHXC4hviUSkxZ3J8ZXNkqwFGyKZVR0TwYCXbprNHl-FYaAHMYYC6esEXNTZ3Jcw7UrS0EnJXOdpulLwKZl1JPz8MlXhLc-I2tQXjblsT_8AicFB7hwB9ygSJSX8vQhfkg9sXaNv-O1KC5iTwgbZovlezjlKY0sIVo2ZDzmulOt9Jl0rrarjm1q4jLlLPelbLYgfFhDkKNcSlgsNneIRRNBouAV94hbz3uUW8yuvmawbSsW1oeWvEIYA15N81pqHZ0NCeHSoj3GEGXNgau9JzMPtY9Zv7o5V3kljP5vfw5UR7uAAXrjdazmtS9BFHmoTEoKO6p-HWsI37vPLfoVQWA0LlPQhNDyOW9w9d1uA9MH92ebCgwvsEQ067ImxyK-T3XGO0v7wRhlsOloNeed9fnKdY2YcjBXyqhJJoiSn56Il_BDJRnwKltt6XP53pI9Su1xY8QQi3qaz0hI1j5fNyBTRQaowVcptPc"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="tender-lamb"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/tender-lamb/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/tender-lamb/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/tender-lamb

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
  cat << EOF > sl.test-y2023m09-w-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=test-y2023m09-w-gss.${ii}
#SBATCH --output=sl.std.out.test-y2023m09-w-gss.${ii}
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
  sbatch sl.test-y2023m09-w-gss.${ii}.sbatch
done
echo "===================================="

