#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/absolute-stag"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/absolute-stag"
export PW_PARENT_NAME="inline.absolute-stag"
export PW_WORKFLOW_NAME="inline.absolute-stag"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.absolute-stag-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWI3ZjAyMjE1NzI3MTkzMjYzZTM1Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYjdmMDIyMTU3MjcxOTMyNjNlMzUiXSwiZXhwIjoxNzg0MDM4NjQwLCJpYXQiOjE3ODE0NDY2NDAsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.JMs9VOoa1mEBMb-2NGoIseJNGbWyuKXz2Ia4iLX3BkKgxLFTuHdsizu_5tjJrfuZ19Flns_b0QNc2Y_kDJ3U__KVaIkUi1Atajc_negSEgZYjzQ4lBCCbK7wwUNzBZCqVwf9MdUCek3YpSqWl9RQ6tvKySrhGNcwICXB1ylSVUwQS8TziBpU354ZtmdkUvhe1rJrBR84TohzIlyoObIJdPG6zsPzixjicAt4H-Qr3azPmz8iCC2TNIk9tXqpYprrjg62dId8QA2eCHEhe3JJt2NVY0BahUb6vcbgyT9aKu4wC0mCcNnjtb6GA8_CGNFioEJM3UkG82y-eF5xPfoLu5WPf4tEA3VGnlfXPOtPh4SRSPca2ZhbGmqXOpQkvyJ-aO26xP2kvKQc8wf0pNLESQqu5i6rYm0q36o9oOvMjhDXcaOutkWdufNNhE-kX2ofVj73Myxj1ypc4PWaAMzTFEICcuSjkm_WixIQyEgCsIR94K4E07JxOL0esg3soXj-hKzDeQR_pcra2alFrxKi8Nb57I-OO-5N4B4GOf8lHaQNKum9eMv1oaRxj2SI5ekwjqpC-vhIN_cO_ssdLEV98bPW_7Mfpcp0bLwxZpvFLPWcDaj_8XfFtyujddNX-lGTXfyS5KQLaUHrmGG1lGYSbQkikELPhxMgT70xBG1v3y0"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="absolute-stag"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/absolute-stag/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/absolute-stag/logs/setup-and-launch/step_6
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/absolute-stag

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
  cat << EOF > sl.Nov-2022-log10-gss.${ii}.sbatch
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=Nov-2022-log10-gss.${ii}
#SBATCH --output=sl.std.out.Nov-2022-log10-gss.${ii}
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
  sbatch sl.Nov-2022-log10-gss.${ii}.sbatch
done
echo "===================================="

