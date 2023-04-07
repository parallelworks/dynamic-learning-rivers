#==========================================
# Merge the physical properties of a site
# (colocated from RiverAtlas) with the 
# chemical properties of a site (from
# ICON_ModEx_ data and collaborators)
# into a single training file and 
# prediction file.
#==========================================

import pandas as pd

# Load input files
train_chem = pd.read_csv('step_01_output_train.csv')
train_phys = pd.read_csv('step_03_output_colocated_train.csv')

predict_chem = pd.read_csv('step_02_output_predict.csv')
predict_phys = pd.read_csv('step_02_output_colocated_predict.csv')

df3 = pd.merge(df1, df2, on='id', how='outer')
