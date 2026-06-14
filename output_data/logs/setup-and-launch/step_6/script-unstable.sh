#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/clean-bass"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/clean-bass"
export PW_PARENT_NAME="inline.clean-bass"
export PW_WORKFLOW_NAME="inline.clean-bass"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.clean-bass-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWM2NGUyMjE1NzI3MTkzMjY0NThlIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYzY0ZTIyMTU3MjcxOTMyNjQ1OGUiXSwiZXhwIjoxNzg0MDQyMzE4LCJpYXQiOjE3ODE0NTAzMTgsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.SI7YRvN6xbcu3gIni8-dvuvrW70LtLBRdarVz9Tr0Wmsicfj7c-xpogD0jBVhIMPS3SHcr_ReDVyXycWeo08Nodb14XGnAP8eXdQCEWqB9aH_uXHMRmCMW0kJApPC0EPh2JG5uKWF0r0JFwMMkkdRd4DOBOn_DKLYRySpFKhJWc71NCsFNaGN6FIvzURaZin0D9Bpn56kjlgyJm3_XvXlUgKVDKwY6ZeXF1iAttWtHbierJSdLShGs-UgVOhqTBk6z-ukcvTv9e1XZn5SbOAX2eLS4rtpzEJz-4tT2FWF--CMdCjaKr2nP65ysONoUvR0fBvaIleFomRWft1_HPjTXZpIXYSaKd3Ewxjq9NOcM76BoRAourFSSayWBojUleebfx49HxOkjBlYNuf4nIFJNlsVRQbVmiLzuo0M-VbBXEtGptx_BoXN-KmoMnvVlYeGkI4GIhQJZxgB4cgURuKMhJJBZWxcC5cFbM4XL1XBbZiAadaKVoBYuu05CnvqU51X3OMfn5u9GMhZfE-MG8tRhjjSnypPYAjR1LKC1dYEUOUwRmkXWl1DGBeOlN6rRFwfsrB7DAZxnUmj2KueGBcwHkdG_Xq1kccFb4ue4yjar5N9LgOnVLm1V0zL8QroseFX0y7Auxn6F2LKXNhjskHQfIXUfKfQfKTl4qjp_AzHHA"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="clean-bass"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/clean-bass/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/clean-bass/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/clean-bass

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
  cat << EOF > sl.Jan-2023-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Jan-2023-log10-gss.${ii}
#SBATCH --output=sl.std.out.Jan-2023-log10-gss.${ii}
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
  sbatch sl.Jan-2023-log10-gss.${ii}.sbatch
done
echo "===================================="

