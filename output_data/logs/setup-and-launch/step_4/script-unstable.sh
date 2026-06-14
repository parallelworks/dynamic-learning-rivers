#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/nice-goose"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/nice-goose"
export PW_PARENT_NAME="inline.nice-goose"
export PW_WORKFLOW_NAME="inline.nice-goose"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.nice-goose-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZjIzMmUyMjE1NzI3MTkzMjY3ZDc5Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJmMjMyZTIyMTU3MjcxOTMyNjdkNzkiXSwiZXhwIjoxNzg0MDY2MDk0LCJpYXQiOjE3ODE0NzQwOTQsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.hRjcBkbhzCnPuwJSPBhRsrXfUflSvThxSIW0LMEslbcpl09Qa7hNbxx0EvOg9MS0Yhif0RfYXkdV2rukRD-brUB4dSsvAaoBysE04EgQ4O8a-zDp-ZQ51qNUI0PkWX3q3c8cpQFXUQ1VrLl3UnyLWvIZh4sN41931BKG62FeGUBZl1UGX7xbdzVXoR57P2RHpWp2HEboTi_Filkc0sayqbu9uHiLTAwUQ1wEAK7gQzLeklFiQZRleNtFRd-SQXyWeLXC7ZUT5IYCxoT-XBsD68b9JnO10tJGloi004fgMibM4_inz18VmrQNqb7IaAuVuRP9ZGpLPhvnIAKTMUsPdO7CE077Y1txwmxDKWjnTUykCrZLHIwYVdoduLjrqu4_uyzxGC8dA5DqOvBcb-Z6RrA5iYxky9p40wMV0FxmMco0SFq5SYjsTpHM0Wp5TVWCxCnYpCAAwhKBKQfNdf4OHthCR0Cx3uncCe97gW3mWSNW0IUvYUE6z3wl98HnceloVH-hyNtrHngMUbSohZSKxS6rpvGJvxrdZ0pAoIYp-qC8mgubO4hIxnAUYITIwXNvR5uU5xMe0iKeyl7QxfofYuwx2DUfiHLKRzMtR4kf5m2uUnpG2bk5VZVBTkXAETtzgZ2BQ_tMqok9jfGRfydZcx9clkfPzdKskrUi0wf6N3c"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="nice-goose"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/nice-goose/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/nice-goose/logs/setup-and-launch/step_4
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/nice-goose

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

