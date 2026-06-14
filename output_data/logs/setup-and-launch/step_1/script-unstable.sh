#!/bin/bash
export CI=true
export PW_USER="pwdemo.stefan"
export PW_PARENT_JOB_DIR="/home/pwdemo.stefan/pw/jobs/lenient-swift"
export PW_JOB_DIR="/home/pwdemo.stefan/pw/jobs/lenient-swift"
export PW_PARENT_NAME="inline.lenient-swift"
export PW_WORKFLOW_NAME="inline.lenient-swift"
export PW_JOB_NUMBER="00001"
export PW_JOBS_DIR="/home/pwdemo.stefan/pw/jobs/"
export PW_JOB_ID="inline.lenient-swift-00001"
export PW_API_KEY="eyJhbGciOiJSUzI1NiIsImtpZCI6InUxY21pS0pvNEI2LXVkY0xtZEJ3dUtMZHlaaDY2dF8xTXptVTFYNUw3aDgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJ3b3JrZmxvdy1ydW46NmEyZWE3NzRjZDE4NmRmNjljZmJkZWZjIiwic3ViIjoidXNlcjpwd2RlbW8uc3RlZmFuIiwiYXVkIjpbIndvcmtmbG93LXJ1bjo2YTJlYTc3NGNkMTg2ZGY2OWNmYmRlZmMiXSwiZXhwIjoxNzg0MDM0NDIwLCJpYXQiOjE3ODE0NDI0MjAsInBsYXRmb3JtX2hvc3QiOiJhY3RpdmF0ZS5wYXJhbGxlbC53b3JrcyIsImdyb3VwcyI6W10sImZlYXR1cmVzIjpudWxsfQ.xJiVLwl94Ye2V9ndP8X5wWjVBNBSHu-Ko7oGDrchDL1KEcba0QAwDfHSW-XCmZROTVuC6beQoPBSJ2slFMwLGiCCNwN8eiNPwmXknvZ7B4sY3DXPB14f6-YmcMScxM94cJPgUfVPr2NUx5YwTJ3YDLU73stqudHHkpWUZnYtiVeYKaNaga3DnhjAF7q3jqa99FD3WFv5ObYPD35uhWVTEvyJEK1nxOrnavdby2uG-xB1n05rfbAWKjUpSfwK-fJXFcSRZ5oURQ9_h5JfrhpgvZQAFGnBkt2qYcJVEdNqBxjRByjs8-vPtfZIJcCZ8J8U8JmpOUqbys-G-v3uJ_onqTIsfOXKkda_c_phyp83hJe2l0B8pcSR4a8Cev2jMzwfVueNKKi4ie5MwHPbrpxfZuMPN5gzzHQRDBcEck7FPjex659eAvCt01DlG-4qSPy66ywp2SwOyy5q39nbggfPBdNcKiG3SDe0lzYlgdQ5IM4zldZNmqVqgT4dKTUQdRPYp3OO0anMDDvuhZc_agh-AUdTIdnUvh9SNiKIB4KwtgZpkK5ulfcBFF8H7rCMIJ3soliYCaAJlB5ka1P9xLAFcA9Tqr_mKqwr_v6mhqWkfzufRQRd5tedd7p-Vg5TkFpkPqp_Uw8ttHK504zDMp5xIhUhlHvJrFDjRWBCo4T4M0s"
export PW_PLATFORM_HOST="activate.parallel.works"
export PW_RUN_SLUG="lenient-swift"
export PW_WORKFLOW_STEP_CURRENT_RETRY=0
export PW_WORKFLOW_STEP_MAX_RETRIES=0
export OUTPUTS="/home/pwdemo.stefan/pw/jobs/lenient-swift/logs/setup-and-launch/outputs-unstable"
cd /home/pwdemo.stefan/pw/jobs/lenient-swift/logs/setup-and-launch/step_1
echo $$ > step.pid
cd /home/pwdemo.stefan/pw/jobs/lenient-swift

mkdir -p ~/bin
rm -rf ~/bin/gh*
wget https://github.com/cli/cli/releases/download/v2.93.0/gh_2.93.0_linux_amd64.tar.gz -O gh.tar.gz
tar -xzf gh.tar.gz -C ~/bin
rm gh.tar.gz
cd ~/bin
mv gh* gh

