#!/bin/bash
#=========================
# We have run a bunch of 
# iterations but now we need
# to rerun them all due to
# the need to test the addition
# of GroupShuffleSplit
#=========================

# Authenticate to GitHub CLI
# via PAT before running this
# script. An example of this setup
# is in the Install GitHub CLI
# and Setup GitHub CLI and git
# steps in:
# https://github.com/parallelworks/sl_core/blob/main/workflow.yaml
# Technically, this whole re-run
# could be done via a GitHub runner,
# but I expect the re-run to take
# longer than the maximum lifetime of
# a runner, so run it as a bash script
# from a laptop instead.

mkdir -p ${HOME}/re-run-tmp-work
cd ${HOME}/re-run-tmp-work
echo Running in $PWD sandbox dir

# Clone the repo (do this once in advance)
git clone https://github.com/parallelworks/dynamic-learning-rivers
cd dynamic-learning-rivers
echo Now in $PWD
git pull

# Define the branches we want to rerun
#list_branches=( \
#    "test-y2023m09-w-log" \
#    "Summer-2019-log10" )

list_branches=( \
    'Dec-2021a-log10' \
    'Dec-2021b-log10' \
    'Jul-2022-log10' \
    'Aug-2022-log10' \
    'Sep-2022-log10' \
    'Oct-2022-log10' \
    'Nov-2022-log10' \
    'Dec-2022-log10' \
    'Jan-2023-log10' \
    'Feb-2023-log10' \
    'Mar-2023-log10' \
    'Apr-2023-log10' \
    'May-2023-log10' \
    'June-2023-log10' \
    'Jul-2023-log10' \
    'August-2023-log10' \
    'Sep-2023-log10' \
    'Oct-2023-log10' \
    'Nov-2023-log10-DO-update-correct')

# For each branch to re-run:
for rerun_branch in "${list_branches[@]}"; do
    echo "Current branch: $rerun_branch"

    # Checkout
    git checkout $rerun_branch
    git pull

    # Copy input files out of repo
    cp -iv ./input_data/ICON-ModEx_*.csv ../

    # Define output branch name with gss
    # ending for GroupShuffleSplit. Only keep
    # first three strings of rerun_branch
    # e.g. <month>-<year>-log10 becomes 
    # <month>-<year>-log10-gss
    new_branch=$(echo $rerun_branch | awk -F- '{OFS="-"; print $1,$2,$3,"gss"}')
    
    # Create output branch from main
    git checkout main
    git pull
    echo Creating new branch $new_branch
    git branch $new_branch main

    # Checkout output branch
    git checkout $new_branch
    git pull

    # Move input files from prevous branch into input_data, overwriting.
    mv -vf ../ICON-ModEx_*.csv ./input_data/

    # Add, commit, and push. This launches the GitHub Action which start workflow on platform
    git add .
    git commit -m "Add input_files for $new_branch at $(date)"
    git push origin $new_branch

    # Force loop to wait until the GitHub Action (and the workflow) is complete
    echo Wait 60 s for Action to start running...
    sleep 60
    run_id=$(gh run list --repo parallelworks/dynamic-learning-rivers --workflow .github/workflows/main.yml --limit 1 --json databaseId -q '.[0].databaseId')
    echo Waiting on run ID $run_id for another 60 s...
    sleep 60
    echo Starting gh run watch...
    gh run watch "$run_id" --repo parallelworks/dynamic-learning-rivers --interval 30 --exit-status | cat
    echo Done waiting on run ID $run_id
    echo Moving to next branch...
done


