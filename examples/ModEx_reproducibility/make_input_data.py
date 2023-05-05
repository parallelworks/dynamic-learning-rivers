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
src_file="ICON-ModEx_Data.csv"

# File that sets the limit of data in the output
ref_file="ICON_ModEx_Data_2022-07-01.csv"
#ref_file="ICON_ModEx_Data_2022-07-29.csv"
#ref_file="ICON_ModEx_Data_2022-09-02.csv"
#ref_file="ICON_ModEx_Data_2022-09-30.csv"
#ref_file="ICON_ModEx_Data_2022-11-02.csv"
#ref_file="ICON_ModEx_Data_2022-12-06.csv"
#ref_file="ICON_ModEx_Data_2023-01-02.csv"
#ref_file="ICON_ModEx_Data_2023-01-27.csv"
#ref_file="ICON_ModEx_Data_2023-02-27.csv"

# Output
out_file=src_file+".out"

#=======================
# Do it
#=======================

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
print(search_str)

# Grab all the data we want by Sample_ID
out_S19S=src[src['Sample_ID'].str.contains(search_str)]

# Write output
out_S19S.to_csv(out_file,mode="w")

