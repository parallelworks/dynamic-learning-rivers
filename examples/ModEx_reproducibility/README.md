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
the first fully automated iteration in April 2023.
This data is special because it is also the
first time the S91S (summer 2019) data were
included in the **same file** as the 2022-2023
data.  The manual ModEx iterations had to merge
S19S with 2022-2023 data from separate files.

`make_input_data.py` uses the April 2023 data
as a reference point to compare with the
`ICON_ModEx_Data_<YYYY>_<MM>_<DD>.csv` files.
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
corresponding files to each `ICON_ModEx_Data_<YYYY>_<MM>_<DD>.csv`
but in the format expected by the fully automated
workflow.

The fully automated workflow was then run for
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
