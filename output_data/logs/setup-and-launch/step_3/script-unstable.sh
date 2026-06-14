#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/amused-glowworm"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/amused-glowworm"
export PW_PARENT_NAME="inline.amused-glowworm"
export PW_WORKFLOW_NAME="inline.amused-glowworm"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.amused-glowworm-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWFjNzljODk5MzVlNGEwYWNlNjYzIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYWM3OWM4OTkzNWU0YTBhY2U2NjMiXSwiZXhwIjoxNzg0MDM1NzA1LCJpYXQiOjE3ODE0NDM3MDUsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.XUZD7ZCB1NgXi0cxZmfcUIIzqYZV76SWl6Qn1knmo6_iZE5uAnBuibecJ_5yJzwPPub_mfUSlzfpqtmBdaXiG1bpD9T_RroJqrd653_vOVc76EuEQzOZHruT6LO6rYSW7CSMIRHEPGjln3EzmHtOU6ZaRzuvQ4zCeEN9sZ_Ulod4rKdqbNIJBRAFDuUepl3_h6wcMFiIMreDb18-yip4vJwJmzi41WQRkyNtMU7MEB9_C4B-QRnJTgL067w0Yu7alxD61WyEHOlKDz0CCpeaAwlw2sisbdjEmBxayqtI7Urxw4ATdespJAYAusGD3XanYOSWspN_UImgVvkMlrShdQLhzgg-_ShS5pLBHaPfYcn04yCdKzhHQJ7W9JD5SS0TWs45_Bf8ndu_ZzFsfdPdb0aacZupIEmbGs1EE9tHo_M0oVwMCGOXjIicwZWYoPeSIPnGSc3JIRhSycf0YA8Lwk2_Eqjy8-JxTf0Zn7j-DJooA-qQB3cHyY-HKwQSvo_CLLqy6HlYclBRHUI-aMFVRRhJmu9qziUnRCOUY-kl1mryVq2LpDOKn2u9o6lmVrVlZbPdMc-mfsupzfmGDRGpSwm71xtxmrwNHb_U8CBb17nWoSy7e1rrVH8Y5MHKA2U5K1gMrZdDIHoQ6N2nzDRFEbxS3v_ZZocXWSEIYguDGJE"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="amused-glowworm"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/amused-glowworm/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/amused-glowworm/logs/setup-and-launch/step_3
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/amused-glowworm

echo Changing to top level working directory
cd ${HOME}
pwd
echo Cloning archive repository...
git clone https://github.com/parallelworks/dynamic-learning-rivers
echo Adding/Checking out archive branch...
cd $(basename https://github.com/parallelworks/dynamic-learning-rivers )
echo Run git fetch
git fetch origin Sep-2022-log10-gss 2>/dev/null || true
if git ls-remote --exit-code --heads origin Sep-2022-log10-gss >/dev/null 2>&1; then
  # exists remotely: check it out tracking origin, fast-forward to latest
  git checkout -B "Sep-2022-log10-gss" --track "origin/Sep-2022-log10-gss"
else
  # doesn't exist: create it fresh from current main
  git checkout -b "Sep-2022-log10-gss" main
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

