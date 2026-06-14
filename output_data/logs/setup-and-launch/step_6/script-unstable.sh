#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/exciting-liger"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/exciting-liger"
export PW_PARENT_NAME="inline.exciting-liger"
export PW_WORKFLOW_NAME="inline.exciting-liger"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.exciting-liger-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWQ1ODRjNGUyNjRmZTUxODM0NDUwIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlZDU4NGM0ZTI2NGZlNTE4MzQ0NTAiXSwiZXhwIjoxNzg0MDQ2MjEyLCJpYXQiOjE3ODE0NTQyMTIsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.RLgrsXtSJXPNK2AxfkZ9aTKzo3CCVhUQWUl1f86UQ_m5RpzN4JMvlbdbU9V97WxWiaqfBrLn0vB7AMFf1OBmjSYmHa0N-tWMu4J3MdGT2Pv4F8FEwCwu9nafDZaTT4QZJ6ZoSmopso9yFe2FvE-gE_SUPfaXVUuuqPVGFqwy17mI0sQ5yMNYxyG582rrChqY35RYZKMIqehF8gdad6-qRPqF3vE1k7JzY9DYkn4bppKib5GO_iqvDOq3BEu2jNyYvITgZEl-o7EGf-UG6qA5UFYKJahifezQH1M-dmUVUtUJY4eywSHLrLHn2PqSG2tX9JzITNA-m2oy4apsQ9jUFDIQTMk7fAoLSHwtnZQ8xBvUHlYRQZp3tVxY2feIwn7wF8iqPq_zzCNKJZzaadKVfXbsvr0aGju0-4xdSSlcunfm7DtXLQUTQEz9qyK7luLTYztrUlO60570LPLGQAbjhmcrmia13WM2VN1yW0QzIleXqq3SbjMgtqHZALcVpRUoW5IT7XVXlwaPOwv1r7RdKj3j90qULbVTRsFFohUbRGtskirF0mG9FAYnDLMLq_UEVXQ16Q4BiSdjs0QY4yHxv2TgkpEALeEKohVvLmTIH_ot-0ieTnpFSMJmJTZypFZ0VS8iRDeL0Inql9p6X9CzOf-mV9mPnmdb0GLKRr3mA1U"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="exciting-liger"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/exciting-liger/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/exciting-liger/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/exciting-liger

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
  cat << EOF > sl.Mar-2023-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Mar-2023-log10-gss.${ii}
#SBATCH --output=sl.std.out.Mar-2023-log10-gss.${ii}
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
  sbatch sl.Mar-2023-log10-gss.${ii}.sbatch
done
echo "===================================="

