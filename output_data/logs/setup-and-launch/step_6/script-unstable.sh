#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/settling-owl"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/settling-owl"
export PW_PARENT_NAME="inline.settling-owl"
export PW_WORKFLOW_NAME="inline.settling-owl"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.settling-owl-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWNkYzAyMjE1NzI3MTkzMjY0OTQyIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlY2RjMDIyMTU3MjcxOTMyNjQ5NDIiXSwiZXhwIjoxNzg0MDQ0MjI0LCJpYXQiOjE3ODE0NTIyMjQsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.1040ii1x9nxtrzU_AnOEb0_KyLq4gnxo2jLgHuC_cA9oPgl-0eNDXqnCfP-ZtfXDICNQcoCDrjcP2G9g8ff-FXufxpQUbkIo77TnLGJmtD3ccjc-SqPuNz_bqaj7m6sjHnR2buBmGfFiS22-NHbwXWcpE158Bnbn8AspWZdMREoCdMKl-KbL53YLVw4W-SIbfxIy3Mtd5D-pHp9zpAtqTtExIgP_8mTK3-zbk17jx29mJ-ieIQ8R4myPcyNpScTBsd97hqpW2T2ASvQzgmOvCU6Vbzd4-BCPeTQIXuHtGJ8QF4VQ0G3a7wYFL0p0-ysdnp-1YrsVkB5_rIAtadYjMcILUs3o5GVV5G6re8N4AaQb98aaAXG33D5mLsG0c8r2wqpGqAaEKf8m49xOv8xRkDtqXFiNKSKrXRreAVv4CUDsPLErofW9krcwrGqfNfQ4fWSH2e4AXzkA_dhaIyBTBVn8RnGXkEcVhH1lQWl2bFp2r3IaDTwUrhkUJswcfFhogd_xpPUkujdI1D6cJpcB-u7Cr9leNTt0dGNPjhSfUDAV0ICe1pn6B_BdybNBPI4_tNkmW3MXKi7jrYt2rrXPTjRmzFxLY-GUWLp__lEs7W_NAQbEaQS35_KNxyHXFmlIDQb5x902NYri3tNMXGAvgl12x3GS-fbqzzA7xba8pdM"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="settling-owl"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/settling-owl/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/settling-owl/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/settling-owl

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
  cat << EOF > sl.Feb-2023-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Feb-2023-log10-gss.${ii}
#SBATCH --output=sl.std.out.Feb-2023-log10-gss.${ii}
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
  sbatch sl.Feb-2023-log10-gss.${ii}.sbatch
done
echo "===================================="

