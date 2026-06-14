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
cd /home/pwdemo.stefan/pw/jobs/tender-lamb/logs/setup-and-launch/step_3
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/tender-lamb

echo Changing to top level working directory
cd ${HOME}
pwd
echo Cloning archive repository...
git clone https://github.com/parallelworks/dynamic-learning-rivers
echo Adding/Checking out archive branch...
cd $(basename https://github.com/parallelworks/dynamic-learning-rivers )
git branch test-y2023m09-w-gss
git checkout test-y2023m09-w-gss
# If the branch does not already exist, the following
# will result in an error but the workflow continues.
# The check is included here since under most 
# circumstances it is expected that the branch will already
# exist when pushing to GitHub and the branch will NOT
# already exist when not pushing to GitHub (a manual test run).
if [[ "true" == "true" ]]; then
  echo Set upstream branch in case it exists already...
  git branch --set-upstream-to=origin/test-y2023m09-w-gss test-y2023m09-w-gss
fi
git pull
cd ${HOME}
echo Cloning ML code repository...
git clone https://github.com/parallelworks/sl_core
cd $(basename https://github.com/parallelworks/sl_core )
git pull
cd ${HOME}
echo Cloning ML data repository...
git clone https://github.com/parallelworks/global-river-databases
cd $(basename https://github.com/parallelworks/global-river-databases )
git pull
cd ${HOME}

