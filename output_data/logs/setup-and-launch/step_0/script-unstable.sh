#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/moving-elk"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/moving-elk"
export PW_PARENT_NAME="inline.moving-elk"
export PW_WORKFLOW_NAME="inline.moving-elk"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.moving-elk-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZjBkYjAyMjE1NzI3MTkzMjY2ZTk1Iiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJmMGRiMDIyMTU3MjcxOTMyNjZlOTUiXSwiZXhwIjoxNzg0MDYwNTkyLCJpYXQiOjE3ODE0Njg1OTIsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.VHFvQ1A3TwPepBGTMxpF_TT9uI95t00iRiYRmpqi7bchEvaic-sVwYkU3wsb3LxNe04pJxfQs-0jnmpJqUFl3Vt0g0uz1wWK6-aovqgzl-oCNA-udDVxi8Qe31qzOccqm_FDy0VWQjHS8DYN5f4Qd_r0wf7Dofk_UXjV-EW6LFWZgNt4gTKAvnWUA8ZCbiqkqJea7TVgXaw4cxMcITdNBl2HKixoPYQN2W4CIwfIOZjkgvctnu2ROgMOoguD1xY8GmRcsGEaQdDNMxUxVxeJaeQDzVpA-ktVCplUtPI0I0VD3-fl0xJi6Mtujp9b7iQ1bzjrG4_Ot2C4b-8ZehnFdLMtQqg3UU1-ftLGmNffIpZxwvrOZZuRmz92Y2WXWgUYKbt_UXl2h3UqWKoKXZ7F0oZOEE0kcc-L05f4UOP6YEKq4qWs7B35SdslOrXyu-zjWf6SEG-q5PsxFBS9kBW-RrDYI98FKBbRY131a2ynNpvvDYTHrehWgvpF6wE2_kXehsP5et8iHP4vpnjdeSe6ggpch0M9zAmD5I__clrAc04t4QuXHb2o6OtoPordrKnlYMytYm9F73n8nmVVx8Tmy7Ov0K2luFB23WQR-wa8QRJzhhUsLKfM1ayMHq3iYkF0kDCcxsicVNzF0LWQmKUdNrYQwlzcMUjcR5Fx_3532ZE"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="moving-elk"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/moving-elk/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/moving-elk/logs/setup-and-launch/step_0
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/moving-elk

echo "Print hostname..."
hostname
echo "Print date..."
date
echo "Print pwd..."
pwd
echo "Print local user: ${USER}"

