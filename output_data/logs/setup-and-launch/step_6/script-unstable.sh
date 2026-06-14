#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/exotic-lark"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/exotic-lark"
export PW_PARENT_NAME="inline.exotic-lark"
export PW_WORKFLOW_NAME="inline.exotic-lark"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.exotic-lark-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZjAyNzljNGUyNjRmZTUxODM1ZGI0Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJmMDI3OWM0ZTI2NGZlNTE4MzVkYjQiXSwiZXhwIjoxNzg0MDU3NzIxLCJpYXQiOjE3ODE0NjU3MjEsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.vgLqYad4X8gVQ2SiKsVB2_7lCTwCtRQ1qLtBW_JBCMow1ThkcXGBDlkcmJ4F-hVXX8nV6rdPNOk6uTk3rlCojyMyckS8i7lMhx0VjYyI7Tm8ASWcuyG-Gp3KleePaSi-yOTpMH_XTsRVqDN5KmDSKXETIYVwxXQGEdHWOO5eiTHvZLEeo0RkkYm6FrmslVOJH4XmJQgR5hSOBrFYwjxSnI9klavmXsmGSE38Qct0JR3CtqkYzKSbzpXPKMRVErDbN3i2URXCESGBEg3T3yAeuPNFyvNNG595dSZN_FePcg1dANg700npCdM7jpoLZAoAQ2POm_xY9XZkVbf0LIjsrNvF5SFHmbbC214v5GeiIAMGCGSSISYJYXImq8z64DDoSZ7Ms0y8LpaDs4EVGfP6UhI2WhSCSF63iJ0Er8JdfylXgrsgNW1xLOcr79SmavaXMW_S21GXVrkGgTSSz-iTvl1tHCicA8oDvAedXTaKL44K1ZpRUNM7nojoivZsJSnm8SrL_OTqI-MEDhp1mLw2KfUoJhRH6bam6n-2qkAoLFRekFybvtDeIylKVqgEhCYyjkjlaTjHodF_LwITujTZ1viwshz_EdLtZkYlx0DEGSKzfBxNxgKtT0-yzfipgeS0rBGbq4j-UJoYeu5tXdPL9u6wawu9XhDkgq1gbTPykNY"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="exotic-lark"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/exotic-lark/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/exotic-lark/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/exotic-lark

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
  cat << EOF > sl.August-2023-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=August-2023-log10-gss.${ii}
#SBATCH --output=sl.std.out.August-2023-log10-gss.${ii}
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
  sbatch sl.August-2023-log10-gss.${ii}.sbatch
done
echo "===================================="

