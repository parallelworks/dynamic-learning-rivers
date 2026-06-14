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
cd /home/pwdemo.stefan/pw/jobs/exotic-lark/logs/setup-and-launch/step_3
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/exotic-lark

echo Changing to top level working directory
cd ${HOME}
pwd
echo Cloning archive repository...
git clone https://github.com/parallelworks/dynamic-learning-rivers
echo Adding/Checking out archive branch...
cd $(basename https://github.com/parallelworks/dynamic-learning-rivers )
echo Run git fetch
git fetch origin August-2023-log10-gss 2>/dev/null || true
if git ls-remote --exit-code --heads origin August-2023-log10-gss >/dev/null 2>&1; then
  # exists remotely: check it out tracking origin, fast-forward to latest
  git checkout -B "August-2023-log10-gss" --track "origin/August-2023-log10-gss"
else
  # doesn't exist: create it fresh from current main
  git checkout -b "August-2023-log10-gss" main
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

