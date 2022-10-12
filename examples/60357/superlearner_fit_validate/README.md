# SuperLearner Fit Validate

Workflow to run stacked ensemble hyperparameter optimization, fitting, and cross-validation evaluation of machine learning models.

# Installation

1. Clone this repository into a PW workflow directory.
2. Create symbolic links to the workflow definition form, `workflow.xml` and workflow structure file, `workflow.json` in the PW workflow directory.
3. Specify the Resource for this workflow under `Workflows -> Resources`.
4. If you did not specify (in `Resources`) a worker image with a SuperLearner Conda environment pre-installed with the required packages, you will need to install the packages listed in `sl_requirements.txt` (provided in this repository) in the default Conda environment on the PW platform.  See the next section below.

# SuperLearner Conda environment

Enter the following commands in the IDE terminal to install the required packages.
These steps will take about 10 minutes.

```bash
# These next lines are commented out b/c only needed if building from scratch.
#conda create --name parsl-pw --clone base
#conda activate parsl-pw
#cd /swift-pw-bin/parslab/build
#pip install .

# Install required packages.  Going package-by-package takes less time
# than using --file sl_requirements.txt b/c fewer dependencies to
# cross-check.
while read requirement; do conda install --yes -c conda-forge $requirement; done < sl_requirements.txt

# Pack up the environment for distribution to the worker nodes.
pwpack
```

If the worker image does not have a Conda environment preinstalled (or the path to
the installed Conda environment is incorrectly entered on the form), then the
default `parsl-pw` Conda environment will be copied, unpacked, and used as the
execution environment.

Preinstalling the Conda environment could result in faster runs but currently
`coasters.py` needs to be modified to check for preinstalled Conda in locations
other than `/tmp`.  It is not a good idea to use `/tmp` for preinstalled Conda
because [/tmp is not persistent on GCE images](https://cloud.google.com/container-optimized-os/docs/concepts/disks-and-filesystem).
