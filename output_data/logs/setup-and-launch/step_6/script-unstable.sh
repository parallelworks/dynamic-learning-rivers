#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/moving-elk"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/moving-elk"
export PW_PARENT_NAME="inline.moving-elk"
export PW_WORKFLOW_NAME="inline.moving-elk"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.moving-elk-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZjBkYjAyMjE1NzI3MTkzMjY2ZTk1Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJmMGRiMDIyMTU3MjcxOTMyNjZlOTUiXSwiZXhwIjoxNzg0MDYwNTkyLCJpYXQiOjE3ODE0Njg1OTIsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.VHFvQ1A3TwPepBGTMxpF_TT9uI95t00iRiYRmpqi7bchEvaic-sVwYkU3wsb3LxNe04pJxfQs-0jnmpJqUFl3Vt0g0uz1wWK6-aovqgzl-oCNA-udDVxi8Qe31qzOccqm_FDy0VWQjHS8DYN5f4Qd_r0wf7Dofk_UXjV-EW6LFWZgNt4gTKAvnWUA8ZCbiqkqJea7TVgXaw4cxMcITdNBl2HKixoPYQN2W4CIwfIOZjkgvctnu2ROgMOoguD1xY8GmRcsGEaQdDNMxUxVxeJaeQDzVpA-ktVCplUtPI0I0VD3-fl0xJi6Mtujp9b7iQ1bzjrG4_Ot2C4b-8ZehnFdLMtQqg3UU1-ftLGmNffIpZxwvrOZZuRmz92Y2WXWgUYKbt_UXl2h3UqWKoKXZ7F0oZOEE0kcc-L05f4UOP6YEKq4qWs7B35SdslOrXyu-zjWf6SEG-q5PsxFBS9kBW-RrDYI98FKBbRY131a2ynNpvvDYTHrehWgvpF6wE2_kXehsP5et8iHP4vpnjdeSe6ggpch0M9zAmD5I__clrAc04t4QuXHb2o6OtoPordrKnlYMytYm9F73n8nmVVx8Tmy7Ov0K2luFB23WQR-wa8QRJzhhUsLKfM1ayMHq3iYkF0kDCcxsicVNzF0LWQmKUdNrYQwlzcMUjcR5Fx_3532ZE"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="moving-elk"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/moving-elk/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/moving-elk/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/moving-elk

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
  cat << EOF > sl.Sep-2023-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Sep-2023-log10-gss.${ii}
#SBATCH --output=sl.std.out.Sep-2023-log10-gss.${ii}
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
  sbatch sl.Sep-2023-log10-gss.${ii}.sbatch
done
echo "===================================="

