# dynamic-learning-rivers

Dynamic river model with observation-driven machine learning. This repository
is a machine learning (ML) archive repository; the training data and corresponding 
trained models are stored here.

This ML archive repository is set up to use a [SuperLearner ML workflow repository](https://github.com/parallelworks/sl_core)
to actually implement the training code and workflow. This ML archive repository 
is a the center of an automated ML workflow that kicks off the training of ML models
(with the code in the ML workflow repo) whenever new data is available in this repository.
The presence of new data is determined with a new release of this repository, not just
a push.  Since we want to automate the training and the archiving, the ML workflow will
automatically start with a new release, train the SuperLearner using the data here, and 
then push a commit of trained models back here.  If the automated workflow were started 
with a push, this feedback loop would become unlimited because all archiving pushes 
would start another round of training.

## Contents

1. `containers` holds build files and instructions for building containers.
2. `input_data` training data for the ML models.
3. `ml_models` machine learning models trained on the `input_data`.
4. `examples` files for direct experimentation with the ML model.
5. `test_deploy_key.sh` allows for testing the deploy key of a repository by cloning, making a branch, and pushing changes on that branch back to the repository.

## Automation Setup

1. This repository holds an API key to a PW account as an encrypted secret.
2. A `.github/workflows/main.yaml` has been added here to launch the ML workflow on release
3. An SSH public deploy key has been added to this repository that corresponds to the 
private key on the PW account that corresponds to the API key in #1, above. This allows that 
PW account to push commits back to this repository after the training is complete.
4. The GitHub action defined in `.github/workflows/main.yaml` is defined in a [separate repository](https://github.com/parallelworks/test-workflow-action).
