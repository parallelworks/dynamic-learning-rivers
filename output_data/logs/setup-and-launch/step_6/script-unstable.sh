#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/lenient-swift"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/lenient-swift"
export PW_PARENT_NAME="inline.lenient-swift"
export PW_WORKFLOW_NAME="inline.lenient-swift"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.lenient-swift-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWE3NzRjZDE4NmRmNjljZmJkZWZjIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYTc3NGNkMTg2ZGY2OWNmYmRlZmMiXSwiZXhwIjoxNzg0MDM0NDIwLCJpYXQiOjE3ODE0NDI0MjAsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.xJiVLwl94Ye2V9ndP8X5wWjVBNBSHu-Ko7oGDrchDL1KEcba0QAwDfHSW-XCmZROTVuC6beQoPBSJ2slFMwLGiCCNwN8eiNPwmXknvZ7B4sY3DXPB14f6-YmcMScxM94cJPgUfVPr2NUx5YwTJ3YDLU73stqudHHkpWUZnYtiVeYKaNaga3DnhjAF7q3jqa99FD3WFv5ObYPD35uhWVTEvyJEK1nxOrnavdby2uG-xB1n05rfbAWKjUpSfwK-fJXFcSRZ5oURQ9_h5JfrhpgvZQAFGnBkt2qYcJVEdNqBxjRByjs8-vPtfZIJcCZ8J8U8JmpOUqbys-G-v3uJ_onqTIsfOXKkda_c_phyp83hJe2l0B8pcSR4a8Cev2jMzwfVueNKKi4ie5MwHPbrpxfZuMPN5gzzHQRDBcEck7FPjex659eAvCt01DlG-4qSPy66ywp2SwOyy5q39nbggfPBdNcKiG3SDe0lzYlgdQ5IM4zldZNmqVqgT4dKTUQdRPYp3OO0anMDDvuhZc_agh-AUdTIdnUvh9SNiKIB4KwtgZpkK5ulfcBFF8H7rCMIJ3soliYCaAJlB5ka1P9xLAFcA9Tqr_mKqwr_v6mhqWkfzufRQRd5tedd7p-Vg5TkFpkPqp_Uw8ttHK504zDMp5xIhUhlHvJrFDjRWBCo4T4M0s"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="lenient-swift"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/lenient-swift/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/lenient-swift/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/lenient-swift

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
  cat << EOF > sl.Aug-2022-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Aug-2022-log10-gss.${ii}
#SBATCH --output=sl.std.out.Aug-2022-log10-gss.${ii}
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
  sbatch sl.Aug-2022-log10-gss.${ii}.sbatch
done
echo "===================================="

