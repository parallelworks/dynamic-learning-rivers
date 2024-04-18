import pandas as pd


#=======================
# Remove the data in ICON-ModEx_Data.csv
# that are **not** in ICON_ModEx_Data_<date_string>.
#
# Note that we also want to keep
# the header and all S19S data
# because S19S data were added later
# in the pipeline
#=======================

#=======================
# Setup
#=======================

# Source file (with as much data as needed)
src_file="ICON-ModEx_Data_Nov-2023.csv"

# List of files that sets the limit of data in the output
ref_file_list=[
    "ICON-ModEx_Data_S19S.csv",
    "ICON-ModEx_Data_Jul-2022.csv",
    "ICON-ModEx_Data_Aug-2022.csv",
    "ICON-ModEx_Data_Sep-2022.csv",
    "ICON-ModEx_Data_Oct-2022.csv",
    "ICON-ModEx_Data_Nov-2022.csv",
    "ICON-ModEx_Data_Dec-2022.csv",
    "ICON-ModEx_Data_Jan-2023.csv",
    "ICON-ModEx_Data_Feb-2023.csv",
    "ICON-ModEx_Data_Mar-2023.csv",
    "ICON-ModEx_Data_Apr-2023.csv",
    "ICON-ModEx_Data_May-2023.csv",
    "ICON-ModEx_Data_June-2023.csv",
    "ICON-ModEx_Data_Jul-2023.csv",
    "ICON-ModEx_Data_August-2023.csv",
    "ICON-ModEx_Data_Sep-2023.csv",
    "ICON-ModEx_Data_Oct-2023.csv"
    ]

#=======================
# Do it
#=======================

for ref_file in ref_file_list:

    # Set output file name
    out_file=ref_file+".out"

    # Load data
    src=pd.read_csv(src_file)
    ref=pd.read_csv(ref_file)

    # Use Sample_ID because as noted in
    # https://github.com/parallelworks/dynamic-learning-rivers/blob/main/scripts/prep_01_intake_train.py
    # Sample_ID is a unique identifier.

    # We need to build a string for searching that
    # has the format:
    # <ID>|<ID>|<ID>|...
    # Append each Sample_ID that corresponds to a
    # data point in the ref_file.
    search_list = list(ref['Sample_ID'])

    # Add S19S data (these data were
    # added downstream in the older ModEx iterations)
    search_list.append("S19S")

    # Insert | between each item in list
    search_str="|".join(search_list)

    # Grab all the data we want by Sample_ID
    out=src[src['Sample_ID'].str.contains(search_str)]

    # Write output
    out.to_csv(out_file,mode="w",index=False)

