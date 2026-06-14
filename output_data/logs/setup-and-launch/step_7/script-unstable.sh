#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/major-civet"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/major-civet"
export PW_PARENT_NAME="inline.major-civet"
export PW_WORKFLOW_NAME="inline.major-civet"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.major-civet-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZjE5MGUyMjE1NzI3MTkzMjY3NjE3Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJmMTkwZTIyMTU3MjcxOTMyNjc2MTciXSwiZXhwIjoxNzg0MDYzNTAyLCJpYXQiOjE3ODE0NzE1MDIsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.Sm4aAUVNQMfs3jWHzMjuf4MtOW9Z6c2NfOxsjXs58K2eevKi17TximSIqrf9gtF_0Rn0U_WBGxzAyJdnxqSLrwrCBI-VZA8LNIOncaXk7eVbTOvyWPY0a9RkUtceg2dQUFfor83oUUl4kF_7NYltSXu4gonjfrPym5nmy0biprYnigHuQBSn_xOQ_8XgtL0vZszy0br8xat4EDwlaxQdQO8iKAmnoMMNLiS9f4umBpBud6tpsUEnWMU0aGE6rVmus_kyVefaI_tjLctWl6I-BHx-cbJbBwEDdaa6YJzMfdfDYpelf-HLQ-4HttkZqh_hGHiogLfqtpdElAIe8AGZKnwGTnDY6JnHYRPTOspsJzL7YAlGZLwcETCOiZUzBUd5nV-q6bYPvPel0qBF9POdeFkn8rLnWf3Nwr6ug6c5xfzKGHrUIOtSrxpK6XkqFVYWRPaPU81WzLvAYDryAeVX_oiSf1Nk0DXFHq3h_l7J_bKj6qYk8g_NEOk7uz1qoS3T1w1Wc73iaJz9MDbJ22D6VxH3Cv2hA_PUfcmGNIIOFjD2xsqjUq0bZ4IttZf8gBGurjhWPcFnRgTn47xTCw9Q9S9E2onpXnDgG9mGL_9Upho-FWj9SGPSs751E6iAyN5JOaHRqT-TBP6ljtwPdEeqT1I7ZmHht2h6AA09eqCgXPA"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="major-civet"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/major-civet/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/major-civet/logs/setup-and-launch/step_7
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/major-civet

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

