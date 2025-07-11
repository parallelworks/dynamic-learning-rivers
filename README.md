# dynamic-learning-rivers

Observation-driven machine learning. This repository
is a machine learning (ML) archive repository; the training data and corresponding 
trained models are stored here. Also, ancilliary preprocessing scripts for
some data wrangling are stored here.

This ML archive repository is set up to use a
[SuperLearner ML workflow repository](https://github.com/parallelworks/sl_core)
that holds the training code itself.  The workflow is divided into two stages:
1. a workflow launch, orchestrated by a GitHub action in this repository
(see `.github/workflows/main.yml`) that starts a high performance computing
(i.e. on a cloud cluster) workflow on the Parallel Works ACTIVATE platform and
2. the [HPC workflow itself](https://github.com/parallelworks/sl_core/blob/main/workflow.sh).

This ML archive repository is at the center of an automated ML workflow that
kicks off the training of ML models (with the code in the ML workflow repo) whenever
new data is available in this repository. The presence of new data is determined with
a new release of this repository, not just a push.  Since we want to automate the
training and the archiving, the ML workflow will automatically start with a new release,
train the SuperLearner using the data here, and then push a commit of trained models
back to the archive repository.  If the automated workflow were started 
with a push, this feedback loop would become unlimited because all archiving pushes 
would start another round of training.

## Contents

1. `input_data` training data for the ML models.
2. `ml_models` machine learning models trained on the `input_data`.
3. `examples` files for direct experimentation with the ML model, including
   scripts for setting up "hindcast" runs.
4. `scripts` contains data preprocessing/wrangling/postprocessing scripts and
   intermediate data specific to this data set that bookend the workflow.
5. `outputs` contains selected output files from the workflow.

The ML models in this archive are stored on different branches. Therefore, the
contents of all the directories listed above will change when you change to
different branches. (The exception to this general rule is `examples` which
was established for setting up reproducibility runs with the ML workflow.)

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
Please see the `README.md` in the [SuperLearner workflow](https://github.com/parallelworks/sl_core) for more details.

## Branch, tag (version), and release naming conventions

This ML archive repository tracks the status of inputs and outputs of the ML
workflow as more data become available. Each model-experiment (ModEx) iteration
is treated as a separate branch. This allows the core fabric of this repository
to evolve over time while also "sprouting" distinct ML models that are a snapshot
of a particular ModEx iteration. Some basic guidelines:
+ **Branches** can have human readable names (e.g. mid-April-2023-test).
+ **Tags** (e.g. `v2.2.1`) are assigned to the state of a branch immediately before the machine learning workflow is run. This is because the ML workflow is started when a release is published, and each release gets a tag. `v2` indicates that the workflow is fully automated (`v1` is partial automation). The middle digit (`v2.2`) reflects changes in the `main` branch and is incremented. The last digit indicates the number of ModEx iterations for this particular state of the `main` branch.
+ **Releases** are automatically named using the create release notes button.
