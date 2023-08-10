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

# Set location of where to work
sl_dirs = glob.glob(workdir+"ml_models/sl_*")

# Initialize storage
hold_out = []
ml_output_df_list = []

fpi_results = pd.DataFrame()
fpi_results_std = pd.DataFrame()

for dir in sl_dirs:
    #====================================================
    # Work on the hold-out metrics - SuperLearner score
    # which is overall results (mean and std of
    # overall model scores)
    #====================================================
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
    
    #=======================================================
    # Work on the SuperLearner FPI metrics
    #=======================================================
    fpi_results_csv = pd.read_csv(dir+"/sl_fpi_results_df")
    
    # Initialize storage for features in this model
    feature_list = []
    importance_list = []
    uncertainty_list = []
    
    # For each feature, extract its value from the dataframe where features
    # are grouped and put that value in the dataframe where features are separate
    # for each feature in feature_list.
    for row in range(len(fpi_results_csv)):
        for feature in fpi_results_csv.loc[row,"Feature"].split(','):
            feature_list.append(feature)
            importance_list.append(fpi_results_csv.loc[row,"Avg_Ratiostack0"])
            uncertainty_list.append(fpi_results_csv.loc[row,"Std_Ratiostack0"])
    
    fpi_results_one_sl = pd.DataFrame(feature_list,columns=pd.Index(['Feature']))
    fpi_results_one_sl.insert(1,"FPI_Ratio_"+str(i),importance_list,allow_duplicates=False)
    
    # Update the feature-by-feature dataframe with the feature_list as an Index
    # Now data can be accessed directly by feature name. Drop the Feature column
    # since it is now the index.
    fpi_results_one_sl.index = list(fpi_results_one_sl["Feature"])
    fpi_results_one_sl.pop("Feature")
    
    # Now data can be accessed directly by feature name, e.g.:
    # fpi_results_one_sl.loc['sgr_dk_rav','FPI_Ratio']
    
    # Do exactly the same thing for the standard deviations associated with each FPI ratio
    fpi_results_one_sl_std = pd.DataFrame(feature_list,columns=pd.Index(['Feature']))
    fpi_results_one_sl_std.insert(1,"FPI_Ratio_"+str(i),uncertainty_list,allow_duplicates=False)
    fpi_results_one_sl_std.index = list(fpi_results_one_sl_std["Feature"])
    fpi_results_one_sl_std.pop("Feature")
    
    # Append the results to overall
    if i == 0:
        fpi_results = fpi_results_one_sl
        fpi_results_std = fpi_results_one_sl_std
    else:
        fpi_results = fpi_results.join(fpi_results_one_sl, validate="1:1")
        fpi_results_std = fpi_results_std.join(fpi_results_one_sl_std, validate="1:1")

# Done with looping over SuperLearners, consolidate results

#=============================================
# Compute/write mean and std of hold-out score
#=============================================

print('Writing mean/std hold-out score...')

mean = np.mean(hold_out)
std = np.std(hold_out)

with open('post_01_output_holdout_score.txt', 'w', encoding="utf-8") as f:
    f.write("Avg: "+str(mean)+"\n")
    f.write("Std: "+str(std)+"\n")

#=============================================
# Flatten the predictions at each site and the
# metric at each site.
#=============================================

print('Finding mean/std of predictions...')

ml_output_all_df = pd.concat(ml_output_df_list)
by_id = ml_output_all_df.groupby(ml_output_all_df['Sample_ID'])
df_avg = by_id.mean()
df_std = by_id.std()

df_avg.to_csv('post_01_output_ml_predict_avg.csv',mode='w')
df_std.to_csv('post_01_output_ml_predict_std.csv',mode='w')

#=============================================
# Write a summary of the FPI data and make a
# plot for visualization
#=============================================

print('Writing summary FPI results...')

# Error bars: The std of the FPI iterations are usually 
# very small compared to the ratios themselves (~10%) 
# and the scatter between SuperLearners.  Find the 
# max value over the whole data set for context:
std_over_avg = fpi_results_std/fpi_results
print('Maximum std/avg FPI ratio over all models: '+str(std_over_avg.max().max()))

# FPI summary figure
fig, ax = plt.subplots(figsize=(20,10))

# Rotate feature names so they are legible in plot
plt.xticks(rotation=45, ha='right', fontsize=20)

# Plot all means from all SuperLearners
ax.plot(fpi_results,'k+')

# Plot the mean and std envelope
ax.plot(fpi_results.mean(axis=1),'k',linewidth=3)
ax.plot(fpi_results.mean(axis=1)+fpi_results.std(axis=1),'k--',linewidth=3)
ax.plot(fpi_results.mean(axis=1)-fpi_results.std(axis=1),'k--',linewidth=3)

# Labels, etc.
ax.grid()
plt.ylabel('FPI improvement ratio', fontsize=20)
plt.xlabel('Feature name')
plt.savefig('FPI.png')

