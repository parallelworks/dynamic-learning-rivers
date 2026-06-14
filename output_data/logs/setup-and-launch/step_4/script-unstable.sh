#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/sweet-eel"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/sweet-eel"
export PW_PARENT_NAME="inline.sweet-eel"
export PW_WORKFLOW_NAME="inline.sweet-eel"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.sweet-eel-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZTliMzhjODk5MzVlNGEwYWNkZjI4Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlOWIzOGM4OTkzNWU0YTBhY2RmMjgiXSwiZXhwIjoxNzg0MDMxMjg4LCJpYXQiOjE3ODE0MzkyODgsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.Da9IJT2R3kVsEk183Cm4BDWsB01LTjvfEHbUIscSLEfC2UOEJLlIPKC_AupDaxbFx_uoLnMCoAPXnjPgx83eqfjfjWinnubQY3ayFVoWzoXY0jk1an4BbE47C096oG7AkJ1ur5UJJrsloNaIlA7qqhSWHSI_UGd_IlehR_0no66Q47InMD8rByCHxi7CyINJrQjNx22Ie8W8oZVYiNujTd15PE3umg34MFKUb3LdmYiULfgrcnPJB_JUnQIiHuKZd_ZcVtYZE8W_Wu-lv_M1egt_5W_0x9WT70Jvjn22N_CMFc-8zH-vAWbilqRpWY1jagTP6k5wnm6182dW6DGeYH9gbZWKsuCfNkWKMrJxxeUmZlf-9ojeXveVVDjH0KbFJnRoYzXNuCl-CuaTysfbO5bF7jIvhEKHiSV9Tm7mVwQeLbMw0C7Jjttm94e9sP-YXD9uQOThSe3vjpfmYbP9_zf4TNyqFAcuV8bXie1zvLyfWiEomev2M_CZKDYcdphcqL03x0xnj2HtGmu6R65o-AjsY0czpoAQKt-_5ThniOpTN9c_zrW1vF8D6sTrjCFcl5PHElIrXqbwnv1kXFhaiMOUJksIEIAcL0MpOJu4reHh-sj2LNIryO5FphRNm7nNVTRb5RKsGyZFEGaabZYfuN0cqwT7wHIrGZyZTnaUQwU"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="sweet-eel"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/sweet-eel/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/sweet-eel/logs/setup-and-launch/step_4
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/sweet-eel

echo "======> Test for presence of Conda environment"
ls /home/${USER}/.miniconda3
if [ $? -ne 0 ]; then
  echo "======> No Conda found; install Conda environment for SuperLearner."
  echo This process can take several minutes.
  cd ${HOME}
  cd $(basename https://github.com/parallelworks/sl_core )
  ./create_conda_env.sh /home/${USER}/.miniconda3 superlearner
else
  echo "======> Conda found!  Assuming no need to install."
fi

