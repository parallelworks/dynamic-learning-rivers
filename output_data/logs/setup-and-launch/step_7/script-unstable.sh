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
cd /home/pwdemo.stefan/pw/jobs/absolute-stag/logs/setup-and-launch/step_7
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/absolute-stag

n_squeue="2"
squeue_wait=10
while [ $n_squeue -gt 1 ]
do
  # Wait first - sbatch launches may take
  # a few seconds to register on squeue!
  echo "Monitor waiting ${squeue_wait} seconds..."
  sleep $squeue_wait
  n_squeue=$(squeue | wc -l )
  echo "Found ${n_squeue} lines in squeue."
done
echo "No more pending jobs in squeue."

