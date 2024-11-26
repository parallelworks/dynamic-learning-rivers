# ModEx Reproducibility

## Hindcast runs

This directory documents the steps taken to
verify the reproducibility of the initial,
manual, ModEx iterations with the new,
fully automated workflow. It is a "hindcast"
of what ML models would have been trained
had the fully automated workflow been
available from July 2022. (Full automation
started in April 2023.)

The `ICON_ModEx_Data_<YYYY>_<MM>_<DD>.csv`
files are the original source files for the
manual ModEx iterations.

`ICON-ModEx_Data.csv` is the data used for
the first fully automated iteration in April 2023
and at the end of 2023, once more data became available,
this file was copied to `ICON-ModEx_Data_Apr-2023.csv`.
The Nov-2023 data is used as the reference point
since numerous small data quality control and
corrections were made between April and November
2023.

Apr-2023 was special because it is also the
first time the S91S (summer 2019) data were
included in the **same file** as the 2022-2023
data.  Previously, the manual ModEx iterations 
had to merge S19S with 2022-2023 data from 
separate files.

`make_input_data.py` uses the Nov-2023 data
as a reference point to compare with the
`ICON_ModEx_Data_<MMM>-<YYYY>.csv` files.
Since minor (i.e. not used in learning) features
were added to the input data as we streamlined
the iterations, only the presence of the `Sample_ID`
in the historical files is used - the presence of
that `Sample_ID` is used to pull out matching lines in
the reference file to reconstruct an input file
that is consistent with the expected format but
also has only the data that were used at the time
of the historical iteration.

`ICON-ModEx_Data.csv.out.<MMM>-<YYYY>` are the
corresponding files to each `ICON_ModEx_Data_<MMM>_<YYYY>.csv`
but in the format expected by the fully automated
workflow.

The fully automated workflow was then run for:
+ `Nov-2023`
+ `Oct-2023`
+ `Sep-2023`
+ `August-2023`
+ `Jul-2023`
+ `June-2023`
+ `May-2023`
+ `Apr-2023`
+ ----------Initial runs here repeated again
+ `Jan-2023`
+ `Feb-2023`
+ `Mar-2033`
+ `Dec-2022`
+ `Nov-2022`
+ `Oct-2022`
+ `Sep-2022`
+ `Aug-2022`
+ `Jul-2022`
and corresponding branches were created in
this GitHub repository.

In retrospect, we saw that that log10 `TransformedTargetRegressor`
helped generate ML models with less bias than the standard SuperLearner
configuration, so we repeated all the runs with `-log10` designation
for consistency and to identify any potential plateauing of model
score/bias reduction as more data are added.

## ModEx methodology test

To explore the extent to which the site selection process of the 
ICON-ModEx approach helps train ML models, we decided to train ML
models with the "highest priority" (HP) and "lowest priority" (LP) 
data and see if each series of ML models' scores improved more or 
less quickly relative to each other.

To ensure an apples-to-apples test, we want to ensure that
the ML models are predicting on the same data set. So, we train
a series of ML models with the HP (LP) 100, 200, and 300 
points. The data points not used by **either** the HP and LP
case ML model training then become the data used for assessing
the model scores.

To implement this, we will:
1. Train each of the 6 ML model ensembles by setting the training data
   to a subset of the 100, 200, or 300 HP or LP points.
2. Reuse/reload each ML models and have it run a [score](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor.score)
   on the unused "neutral" data. To simplify the process,
   we use the "neutral" data as the collaborator site data.

So, first, I need a list of HP and LP sites. Use the output
from the Nov-2023-log10-DO-update-correct branch since this ML model
was trained on all the available and corrected data. Copy that file
here for archiving from ./scripts/post_01_output_ml_predict_avg.csv.

This process is coordinated by the following three notebooks:
+ `get_hp_and_lp_sites.ipynb` - rank HP/LP sites by combined.metric = pca.dist * est.error
+ `get_hp_and_lp_sites_by_pca_only.ipynb` - rank HP/LP sites by pca.dist only to remove the potential bias due to the magnitude of the respiration rate.
+ `get_cold_and_hot_spot_sites.ipynb` - rank sites by magnitude of respiration rate - accentuates the potential bias due to the magnitude of the respiration rate.

After the notebooks run, there is some manual work necessary with
`head` and `tail` to get the 100, 200, and 300 ranked sites as 
described in the notebooks. The results are stored in the separate
subdirectories here for use by the ML workflow.
