# dynamic-learning-rivers

Dynamic river model with observation-driven machine learning. This repository
is a machine learning (ML) archive repository; the training data and corresponding 
trained models are stored here. Also, ancilliary preprocessing scripts for
some data wrangling are stored here.

This ML archive repository is set up to use a [SuperLearner ML workflow repository](https://github.com/parallelworks/sl_core)
that holds the training code itself.  The workflow is divided into two stages:
1. a workflow launch, orchestrated by a GitHub action in this repository (see `.github/workflows/main.yml`) that starts a high performance computing (i.e. on a cloud cluster) workflow on the PW platform and
2. the [HPC workflow itself](https://github.com/parallelworks/sl_core/blob/main/workflow.sh).
Therefore, this ML archive repository is at the center of an automated ML workflow that
kicks off the training of ML models (with the code in the ML workflow repo) whenever
new data is available in this repository. The presence of new data is determined with
a new release of this repository, not just a push.  Since we want to automate the
training and the archiving, the ML workflow will automatically start with a new release,
train the SuperLearner using the data here, and then push a commit of trained models
back to the archive repository.  If the automated workflow were started 
with a push, this feedback loop would become unlimited because all archiving pushes 
would start another round of training.

## Contents

1. `containers` holds build files and instructions for building containers.
2. `input_data` training data for the ML models.
3. `ml_models` machine learning models trained on the `input_data`.
4. `examples` files for direct experimentation with the ML model.
5. `test_deploy_key.sh` allows for testing the deploy key of a repository by cloning, making a branch, and pushing changes on that branch back to the repository.
6. `scripts` contains data preprocessing/wrangling/postprocessing scripts specific to this data set that bookend the workflow.

## Automation Setup

1. This repository holds an API key to a PW account as an encrypted secret.
2. A `.github/workflows/main.yml` has been added here to launch the ML workflow on release
3. An SSH public deploy key has been added to this repository that corresponds to the 
private key on the PW account that corresponds to the API key in #1, above. This allows that 
PW account to push commits back to this repository after the training is complete.
4. The GitHub action itself is defined in `action.yml` in a [separate repository](https://github.com/parallelworks/test-workflow-action) - this action starts a resource on the PW platform, launches the HPC workflow (sending workflow-parameters from this repository's launch workflow to the HPC workflow on PW).

## Installation

There's no manual installation assocaited with this repository since the running
code is already setup in `.github/workflows/main.yml`. However, the code for the
ML workflow (i.e. the SuperLearner) does need to be established on the platform.
Please see teh `README.md` in the [SuperLearner workflow](https://github.com/parallelworks/sl_core) for more details.
