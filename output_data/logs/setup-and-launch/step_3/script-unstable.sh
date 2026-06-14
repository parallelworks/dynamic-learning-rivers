#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/fast-loon"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/fast-loon"
export PW_PARENT_NAME="inline.fast-loon"
export PW_WORKFLOW_NAME="inline.fast-loon"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.fast-loon-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZTlmN2VjODk5MzVlNGEwYWNlMGU1Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlOWY3ZWM4OTkzNWU0YTBhY2UwZTUiXSwiZXhwIjoxNzg0MDMyMzgyLCJpYXQiOjE3ODE0NDAzODIsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.uW8lE7ZkTCYv0LPuZpSudBi0z4mLUI-EbkL7hbfrqB1Ws16ez3I6tQmqeqLSq0PYUH2i8tL0ROznXT6wZwn1fbb7FLwd9Xvi_edj_qv8wAIngSrHCFnQMys_w4gftGszVRc6E32c384s42_X1Bk0NE8mTA1M4M_nPJyQ15DloWk_7tfiZnklnZFTxs7-dXneROSzq1MyTiyaix-SnzRIeDuHlJPlM9tqxVZL0D2pj4v2vBs7_D0hmBROrCPeNwJKY9yArG5IV06d8B0-8zfXSbd_XPeNOpVPkFcWyx9YHF_h1xemrw8XxNi--QSnC58MhsuNqHkoALVvnN3y6xuIsrfy2kUgMI4L8H506BlxkWKLbvL15-FOrs8JMgpktqM0GXabX9SGxtVHUaPRRwl86-3LS-3crG7U5GFZb2rvlFxrOto6aGkKI8SIzJwZ7KQG-jyIkjM2e_EzhD3DGFMPmRqjf3-3I3c04yW0BWdu89tB4oKD6yI7pzpi8BDlJDTf2XLyaXRHFBUdcUOU5uZqb7frxbc1TtfQ2yM2Qr6o_lMsIoYDP77vCUHm61R_1dJS5sBVVeqLI-hNMXErJVn6tIc5--W-Yxie0GjVWpRXMcW0kTHG0FlIaJRUc2dgCeS8Tt8ajrk6oMmnnXtCxAZNPaeXGwNbWsau3kv7nznwEM0"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="fast-loon"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/fast-loon/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/fast-loon/logs/setup-and-launch/step_3
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/fast-loon

echo Changing to top level working directory
cd ${HOME}
pwd
echo Cloning archive repository...
git clone https://github.com/parallelworks/dynamic-learning-rivers
echo Adding/Checking out archive branch...
cd $(basename https://github.com/parallelworks/dynamic-learning-rivers )
echo Run git fetch
git fetch origin Dec-2021b-log10-gss 2>/dev/null || true
if git ls-remote --exit-code --heads origin Dec-2021b-log10-gss >/dev/null 2>&1; then
  # exists remotely: check it out tracking origin, fast-forward to latest
  git checkout -B "Dec-2021b-log10-gss" --track "origin/Dec-2021b-log10-gss"
else
  # doesn't exist: create it fresh from current main
  git checkout -b "Dec-2021b-log10-gss" main
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

