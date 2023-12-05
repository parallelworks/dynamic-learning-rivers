# ModEx Reproducibility

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

