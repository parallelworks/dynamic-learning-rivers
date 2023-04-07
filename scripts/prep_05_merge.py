#==========================================
# Merge the physical properties of a site
# (colocated from RiverAtlas) with the 
# chemical properties of a site (from
# ICON_ModEx_ data and collaborators)
# into single training file and 
# single prediction file.
#==========================================

import pandas as pd

# Load input files
# Phys (colocated) files are space separated
train_chem = pd.read_csv('prep_01_output_train.csv')
train_phys = pd.read_csv(
    'prep_03_output_colocated_train.csv',
    sep = ' ')

predict_chem = pd.read_csv('prep_02_output_predict.csv')
predict_phys = pd.read_csv(
    'prep_04_output_colocated_predict.csv',
    sep = ' ')

# Predict chem data is loaded with Site_ID, not Sample_ID
# Rename the column to Sample_ID.
predict_chem.rename(
    columns={"Site_ID": "Sample_ID"},
    inplace=True)

# Do the merge
train_merged = pd.merge(
    train_chem,
    train_phys, 
    on='Sample_ID', 
    how='outer')

predict_merged = pd.merge(
    predict_chem, 
    predict_phys, 
    on='Sample_ID', 
    how='outer')

# Load the larger predict data

# Cut all columns to cols in larger predict data

# Append larger predict to collab predict

# Save cut,merged train and predict
train_merged.to_csv('train.tmp.csv')
predict_merged.to_csv('predict.tmp.csv')
