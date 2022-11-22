#===============================
# Generate a series of files with
# more and more rows based on the
# pca.dist column of sl_pca_training.csv
#==============================

import matplotlib.pyplot as plt
import pandas as pd

#=============================
# Script parameters
#=============================

# Initial size of the first training set
num_init = 100

# Size of subsequent "batches"
num_batch = 74

#============================
# Load dataframe
#============================

all_data = pd.read_csv('sl_pca_training.csv')

# Total number of rows (Pandas automatically
# accounts for the header line)
num_rows = all_data.shape[0]

#============================
# Random shuffle dataframe
#============================ 

all_shuffled = all_data.sample(frac=1).reset_index(drop=True)

#============================
# Sort dataframe by pca.dist
#============================

# We want the highest distances first
all_sorted = all_data.sort_values(by='pca.dist',axis=0,ascending=False).reset_index(drop=True)

#============================
# Plot progression of pca.dist
#============================

fig, ax = plt.subplots()
ax.plot(all_shuffled['pca.dist'],'k.')
ax.plot(all_sorted['pca.dist'],'r.')
ax.legend(['Shuffled','Sorted'])
ax.set_xlabel('Order to add data')
ax.set_ylabel('PCA distance metric')
ax.grid()
plt.savefig('./pca_dist_shuffle_sort.png')

#============================
# Drop pca.dist from all datasets
#============================

all_shuffled.pop('pca.dist')
all_sorted.pop('pca.dist')

#===========================
# Write out each dataframe as
# a series of files that get
# larger and larger
#===========================

# Initial data set
all_shuffled.iloc[0:num_init,:].to_csv('r_1.csv', index=False)
all_sorted.iloc[0:num_init,:].to_csv('o_1.csv', index=False)

# Loop over subsequent data sets
nn = num_init
n_file = 2
while nn <= num_rows:
    nn = nn + num_batch
    if nn > num_rows:
        # We have overrun the end of the file, so simply print the whole DF
        all_shuffled.iloc[0:num_rows,:].to_csv('r_'+str(n_file)+'.csv', index=False)
        all_sorted.iloc[0:num_rows,:].to_csv('o_'+str(n_file)+'.csv', index=False)        
    else:
        # We are still marching down the file
        all_shuffled.iloc[0:nn,:].to_csv('r_'+str(n_file)+'.csv', index=False)
        all_sorted.iloc[0:nn,:].to_csv('o_'+str(n_file)+'.csv', index=False)
    n_file = n_file + 1

print("Done")
