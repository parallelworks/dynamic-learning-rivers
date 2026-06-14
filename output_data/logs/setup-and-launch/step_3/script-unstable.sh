#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/advanced-bobcat"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/advanced-bobcat"
export PW_PARENT_NAME="inline.advanced-bobcat"
export PW_WORKFLOW_NAME="inline.advanced-bobcat"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.advanced-bobcat-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWZiMDdjNGUyNjRmZTUxODM1OGRjIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlZmIwN2M0ZTI2NGZlNTE4MzU4ZGMiXSwiZXhwIjoxNzg0MDU1ODE1LCJpYXQiOjE3ODE0NjM4MTUsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.uw3jE02h7yYw7loycyokeQDamm7KN6DwFCKY_PM_RDfES-An8waEDIvrHzTcObD58Y4Ha_9ofqsQxZQeyabl_nBF_vwh_CkK7XZS-DdwDa81TTDc0TnuhJbE2ZIV9STux9KH9iBpBE7aazBfXjtgrIaB31awQlGeOQ-mEeGIBYb-C3Gx0_Kayn4R6ZiG8HaJJwlNAaifycrQBVDVHzElMgV8i9qRB-BCsskAn_H9fb4G5SBH-Gne24YdiYGDh5GNYrhn2y05QXQ6y4cK6vZe_WVCbDByRSEQh0uaBWyrAQXjqyCk-w_CCuGDFRD-esty52LvCchepU9gksj-gL9QYxAfk5pSrAGSxUXqOcwrk86yaXei0gZe6NeAokmNh4a03Ag3M23xlIOsy5e3o4JAIPzVLXzznkUWEIctdKzTasPgmPVs-a5SZpBWJBTwSIJ3qdxaoqg506zkNr9gSYI88HW6O7bZTmY3aXco1aFg2PoVmYe7JBzUR_n5o13MmdXOPay3N-H72zvEetLFhHDEGfuM3YYSppnP8d36sU90hvWlKgqgJ3-Sle4k5ycvuGVf5s3-vatPpYI2Pm8F5MOANFikGclrQXQUeVuyVfmAhHXj2AdCEmHBnH2boquXVthVSspCRiOTMpox2NMOyvotkTQ2JFcy5EkK8LXpnZviWKc"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="advanced-bobcat"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/advanced-bobcat/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/advanced-bobcat/logs/setup-and-launch/step_3
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/advanced-bobcat

echo Changing to top level working directory
cd ${HOME}
pwd
echo Cloning archive repository...
git clone https://github.com/parallelworks/dynamic-learning-rivers
echo Adding/Checking out archive branch...
cd $(basename https://github.com/parallelworks/dynamic-learning-rivers )
echo Run git fetch
git fetch origin Jul-2023-log10-gss 2>/dev/null || true
if git ls-remote --exit-code --heads origin Jul-2023-log10-gss >/dev/null 2>&1; then
  # exists remotely: check it out tracking origin, fast-forward to latest
  git checkout -B "Jul-2023-log10-gss" --track "origin/Jul-2023-log10-gss"
else
  # doesn't exist: create it fresh from current main
  git checkout -b "Jul-2023-log10-gss" main
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

