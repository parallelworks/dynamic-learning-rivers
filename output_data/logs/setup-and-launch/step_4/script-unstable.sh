#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/engaging-lamb"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/engaging-lamb"
export PW_PARENT_NAME="inline.engaging-lamb"
export PW_WORKFLOW_NAME="inline.engaging-lamb"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.engaging-lamb-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWEzNjljZDE4NmRmNjljZmJkZDZmIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYTM2OWNkMTg2ZGY2OWNmYmRkNmYiXSwiZXhwIjoxNzg0MDMzMzg1LCJpYXQiOjE3ODE0NDEzODUsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.1bIxNEZaAMCYLY1WTs-wcBJQmLKZ_coTPvbpX7CCfPiij0BNYh2G_F4zSUKlloQdkZbxQTPMPMFm3_NraK7MEZLqciGZKWDpqPHhLpM2zSReuLXr33J2Du-6DVdWni9dFXDJSZIb7yRBVp-Eh6agHpqSeMerwhDtnbB9t5C9Ky3CY-0JuOqI5V8gkem80tQX9wNyQg2ugdZ8AvQjwAKKlX_g927dMxw7750g2Ibyx22IP-ANIe7U3ZK4JaTLHTByaqQQRRmVUK4B-2hkhzynJxgCkvk3rXejP05ob3ApfRhoAE7jDz8Nws9wrPv9CKEU1MyW27x_GDkGI2ubGHcwFv77N6w6U0j170rmbx9Qdj_F_XHl-zjXh0j85ZpKdKHTy5oGIX-S5abDhQpQ-l1BwsIbqGsRnJq69BEBj93tAPXwhTlo5le6xbopyGcOjeBb07uDsmGquciknXkrx_Lj6wXBDirrsQ09GBd4f5hDYMjUbhj_LaP3SlXByXXL2aq1OGcpB62eMBRxQVKpIJYg9mqmlYi6PqFY81uzuM3nIoR15ZfR_VbKCLBJDxPbMVlceasxIs1aeVuD6srPory00hONy8KQnqyOYZx7gSYcY2gHGq_Ty7Kg0RGd9toMhSzV-tqTAVRKeIR-mu9p9AQpDPvKn3Jb2FCUXPC7ANVwNNw"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="engaging-lamb"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/engaging-lamb/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/engaging-lamb/logs/setup-and-launch/step_4
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/engaging-lamb

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

