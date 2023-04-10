#==============================
# Load the results from each 
# ML model and compute the mean
# and standard deviation over
# all the models' predictions
# at each site.
#==============================

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

workdir="../"

# Overall results (mean and std of
# overall model scores)
sl_dirs = glob.glob(workdir+"ml_models/sl_*")
hold_out = []
ml_output_df_list = []
for dir in sl_dirs:
    #print(dir)
    with open(dir+"/hold-out-metrics.json", 'r') as in_file:
        # Load hold out metric from JSON file
        raw_hold_out = list(json.load(in_file).values())[0]
            
        # If this value is less than zero, make it zero
        # (Applied to models that do worse than a
        # "persistance forecast")
        if raw_hold_out < 0:
            hold_out.append(0)
        else:
            hold_out.append(raw_hold_out)
        
        # Load main ML workflow output file
        # (predictions and error/metric)
        ml_output_df_list.append(pd.read_csv(dir+"/sl_pca.csv"))

mean = np.mean(hold_out)
std = np.std(hold_out)

with open('post_01_output_holdout_score.txt', 'w', encoding="utf-8") as f:
    f.write("Avg: "+str(mean)+"\n")
    f.write("Std: "+str(std)+"\n")

# Flatten 
ml_output_all_df = pd.concat(ml_output_df_list)
by_id = ml_output_all_df.groupby(ml_output_all_df['Sample_ID'])
df_avg = by_id.mean()
df_std = by_id.std()

df_avg.to_csv('post_01_output_ml_predict_avg.csv',mode='w')
df_std.to_csv('post_01_output_ml_predict_std.csv',mode='w')

