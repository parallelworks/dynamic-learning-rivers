#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/ethical-perch"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/ethical-perch"
export PW_PARENT_NAME="inline.ethical-perch"
export PW_WORKFLOW_NAME="inline.ethical-perch"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.ethical-perch-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWIxYzAyMjE1NzI3MTkzMjYzYjJiIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYjFjMDIyMTU3MjcxOTMyNjNiMmIiXSwiZXhwIjoxNzg0MDM3MDU2LCJpYXQiOjE3ODE0NDUwNTYsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.gcFKxJJletW9lj7hKKszjirf_37hwvQ1Q4XJNGlsDDJBQQL4F3VYSbKp8OyjrvpvJT43m-9ZbCXc1NhXlmjzqm2tE67WXGxoJxxGgNvgtHBm13UJk8k_CDqY9WSsi56wvKygCYnGRqY55oc_g7tDPTqs-1n2bjEdxq8tzKO1aUk5XQK92Fw-vAYBCGtB9g7Pk37VOdYw1oW2y6wnzRpkOenA8ujSZiCSvjIuAwDsKPl6hKETl2pRQKc-aRLEIkDc6xItzR0Iguq5-JkgFdNB0iD0J2WaObM6SR-ELX7Cp9GzUREC-tjiJR5ipeBZmAJO5bG9wwHraqpWZHFgzCVZ8AhWyRguourlt1e8IKSosesQU8oqymM7ft5imJNM8LecOiFCRJYKzuYbtt1QniRyd1n7GztqqxHbmHyWpolcHcaC261i5qScW6i7MGc5nlA2PHU1s2ampAaeF74k-tOjjSMYS0WIQ9obB1YGktuMJcHfqJfRGZTylbL1rZf8ySCrOLwZ8-5jQMqCHYjS630UbroFzZf-kN9FPsaKOuD5wW3nqYkua0qtyCwrJADw0WFU9685TF0exh7Uf2OCtbnAtoh_wEe2p_1I_91MWwEQL4gj3dcmPZUusM4FLfpemjIhapU-MF1W2cVsFjrDJDpDZ0dDpbaURyuIOiIj4gRq8A0"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="ethical-perch"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/ethical-perch/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/ethical-perch/logs/setup-and-launch/step_3
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/ethical-perch

echo Changing to top level working directory
cd ${HOME}
pwd
echo Cloning archive repository...
git clone https://github.com/parallelworks/dynamic-learning-rivers
echo Adding/Checking out archive branch...
cd $(basename https://github.com/parallelworks/dynamic-learning-rivers )
echo Run git fetch
git fetch origin Oct-2022-log10-gss 2>/dev/null || true
if git ls-remote --exit-code --heads origin Oct-2022-log10-gss >/dev/null 2>&1; then
  # exists remotely: check it out tracking origin, fast-forward to latest
  git checkout -B "Oct-2022-log10-gss" --track "origin/Oct-2022-log10-gss"
else
  # doesn't exist: create it fresh from current main
  git checkout -b "Oct-2022-log10-gss" main
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

