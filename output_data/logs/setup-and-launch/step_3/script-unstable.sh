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
cd /home/pwdemo.stefan/pw/jobs/clean-bass/logs/setup-and-launch/step_3
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/clean-bass

echo Changing to top level working directory
cd ${HOME}
pwd
echo Cloning archive repository...
git clone https://github.com/parallelworks/dynamic-learning-rivers
echo Adding/Checking out archive branch...
cd $(basename https://github.com/parallelworks/dynamic-learning-rivers )
echo Run git fetch
git fetch origin Jan-2023-log10-gss 2>/dev/null || true
if git ls-remote --exit-code --heads origin Jan-2023-log10-gss >/dev/null 2>&1; then
  # exists remotely: check it out tracking origin, fast-forward to latest
  git checkout -B "Jan-2023-log10-gss" --track "origin/Jan-2023-log10-gss"
else
  # doesn't exist: create it fresh from current main
  git checkout -b "Jan-2023-log10-gss" main
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

