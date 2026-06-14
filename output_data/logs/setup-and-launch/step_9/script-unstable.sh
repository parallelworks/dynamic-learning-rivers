#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/organic-javelin"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/organic-javelin"
export PW_PARENT_NAME="inline.organic-javelin"
export PW_WORKFLOW_NAME="inline.organic-javelin"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.organic-javelin-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWYwNjQyMjE1NzI3MTkzMjY1YmM2Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlZjA2NDIyMTU3MjcxOTMyNjViYzYiXSwiZXhwIjoxNzg0MDUzMDkyLCJpYXQiOjE3ODE0NjEwOTIsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.cbqKzxlR3irHKCON0nh_4L7VPZBKYp1ScJvASxSCDkT9OwPaTBkns_fqN0fYcOU32ccHEMnB-Cs7CjWcLjj01DwINpGiNoGgXIPq8lRxSwN1KWD3VK5LXUH8Glk71U1bgWM4cQyQwx7Z9ARiS7-YhyEjAwwy0UVdGM_nFB4SzZuom0M7-maotYqJ1xz7CdkI8kKcFd5PktCsupFU-06iNXM0SicztJpJGu_cV61FiFHNKdHvdBa_ceU2d_0eN2AD_UW2ZscmMzTE_E5IQX6ZQcV4KoS-9fkro3BRIQS-9lwJ8CHGqyr8ByB3lMjpbR6fgGl9CMJxslWzWqANCrozDvqP1lzDJJx-_ZDwvtQ4dxZx114HA9dQ5C7m2utv5BNCWuZR-b0uYfj-zEQp8fYLTN6X6ad7V0hbqg4paXFEPtwpB8twtjpz5XgR2HZBmmx_Gwgtb46Q_FCTG3otyBPBpZrP3DDofej_5VuRG09F7AFh_1JdYi_7DzO2747oJ4-V-mZn4BPkDcqb4f28-dd2410irG-sOCSF3VH0ORieVDm3i8xUZO2073PeB7TYDzZ_5-KHxRTJFDswldlKuc4s-Gjk8_bK0AW3A1lTH_yllNKlcMzWFdiV84t-MdGEpZGGrcRXRCC87dGsyH5UtWQvmieYxJtOBclgctOpAJwm7ek"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="organic-javelin"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/organic-javelin/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/organic-javelin/logs/setup-and-launch/step_9
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/organic-javelin

echo "======> Stage workflow logs to repo for committing"
rsync -av ./logs ${HOME}/$(basename https://github.com/parallelworks/dynamic-learning-rivers)/output_data/

