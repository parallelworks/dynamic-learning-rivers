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
num_batch = 50

#============================
# Load dataframe
#============================

all_data = pd.read_csv('sl_pca_training.csv')

#============================
# Random shuffle dataframe
#============================ 

all_shuffled = all_data.sample(frac=1).reset_index(drop=True)
print(all_shuffled)

#============================
# Sort dataframe by pca.dist
#============================

# We want the highest distances first
all_sorted = all_data.sort_values(by='pca.dist',axis=0,ascending=False).reset_index(drop=True)
print(all_sorted)

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

