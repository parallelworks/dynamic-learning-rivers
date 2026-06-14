#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/funky-tiger"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/funky-tiger"
export PW_PARENT_NAME="inline.funky-tiger"
export PW_WORKFLOW_NAME="inline.funky-tiger"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.funky-tiger-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWJlZGYyMjE1NzI3MTkzMjY0MWJjIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYmVkZjIyMTU3MjcxOTMyNjQxYmMiXSwiZXhwIjoxNzg0MDQwNDE1LCJpYXQiOjE3ODE0NDg0MTUsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.f3IXBtXypYRJysqhHVCgC8hb1fNTtoCO4DkKp7rZWWb05eQDI0n6T1CxTrR6XSYMEevxZcWwf_aMrcbiE6GJpJmM_TcaUZXfx8CD_DVZ-LnqdCnnMr2ILx1N_SGblGnLCh-tNxoSHQbBNgzv4XujFCiWFrgVVqgf0sjR5CCJYH-hUSdhWih63I93hKuZzzIVfjHC8xju95HTDwTbHEbiipFexpA5Kcy821Cf_nULV65mGFAbVk9ylHfwmDtOGuVoGD321Jb0uo8atn-g6WJudGx_C17z0HRKjhIDdOi_6OjSnvdy-UBxs8HwnELq7TEBE5dw4rchicPwropvEL68hHgQ2M6m6tcNpXr9JK1tyh9Oalup6OxnR2Xw2YJ6aulm7cNtDpDLMZnWUSQvNHGJ-x2VZslSj_Mglo1W_EMgjKp6S40BGAhISXbB-bthnzGr5RDhUK4xj-Mm2I1woqxGFAxhMDrKxhty6Q21Fzfwroizjq7x_LR6sI4mGBM5AQagUCQiSfj5djRhS0OoKtk8o-xEtOzvhrk5gmJaofI-I7Hg1d3-Jf3mARtdRb8d1y2fHnfU-n-7YYZp9h0AGjOv_U7O7n28QquOLxSh-XaFixv71eMmOYSCJ-jCA2sdBZca394hof0_Vyuzb7btUrAu70qLgZGAyjhU1GVUevUoJPQ"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="funky-tiger"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/funky-tiger/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/funky-tiger/logs/setup-and-launch/step_4
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/funky-tiger

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

