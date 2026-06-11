# Coordinate data intake

import pandas as pd
import numpy as np
import o2sat
import argparse

#print("Parsing input arguments...")
parser = argparse.ArgumentParser()
parsed, unknown = parser.parse_known_args()
for arg in unknown:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg)
        #print(arg)

args = parser.parse_args()
#print(args.__dict__)
target_name = args.target_name


data = pd.read_csv("../input_data/ICON-ModEx_Data.csv")

# List the variables we want here.
# Order of the list sets left-to-right column order in dataframe/csv output
# Sample_Kit_ID is not unique - shared among sites but we keep it here
# for group ID construction for GroupShuffleSplit. We also need to include
# the Site_Name to check for three NEON sites that have different
# Sample_Kit_IDs.
vars_to_use=[
    'Sample_Kit_ID',
    'Sample_ID',
    'Sample_Longitude',
    'Sample_Latitude',
    'Mean_Temp_Deg_C',
    'pH',
    'Mean_DO_mg_per_L',
    'Mean_DO_percent_saturation',
    target_name,
    'Site_Name'
]

# Check that each variable we want to use is actually available
for var in vars_to_use:
    if var in data.columns:
        print('Requested input feature '+var+' is available in data set.')
    else:
        print('Requested input feature '+var+' is NOT in data set. Make fill column.')
        data[var] = np.nan
        print('WARNING: Made fill column. If these fill values are not replaced later, ML training will crash.')

# Grab a view of just the subset we want
core_vars = data[vars_to_use]

# Drop any data points with no respiration rate
# (no target data)
targets = core_vars.dropna(
    axis='index',
    subset=[target_name])

# Drop any data points with no oxygen (both missing)
#targets.dropna(
#    axis='index',
#    how='all',
#    inplace=True,
#    subset=['Mean_DO_mg_per_L','Mean_DO_percent_saturation'])

# Drop any data points with no temperature
#targets.dropna(
#    axis='index',
#    inplace=True,
#    subset=['Mean_Temp_Deg_C'])

# Drop any data points with no pH
#targets.dropna(
#    axis='index',
#    inplace=True,
#    subset=['pH'])

#-------------------------------------------------
# Try to recover as much oxygen data as possible - if there
# is one oxygen value, compute the other. Since we are working 
# with river data and river salinities are usually under 10 PSU, 
# we can assume S=0 and the error in saturated O2 will be less 
# than about 10% over a very wide range of temperatures.  
# Temperature has the biggest impact on saturated O2 in water.
#-------------------------------------------------
# Example using sw_o2sat with sa=0 and whatever temperature 
# is at a given time.  Then, to compute the percent saturated oxygen,
#
# percent_o2sat = 100*o2/sw_o2sat(0.0, temperature)
#-------------------------------------------------
# Units
#
# ICES is a great resource for water units conversions,
# https://ocean.ices.dk/tools/unitconversion.aspx. We 
# can convert the sw_o2sat output (mL/L) to the units 
# used in hydrology (mg/L) with:
#
# o2sat_mg_per_l = o2sat_ml_per_l*1.4291
#
#-------------------------------------------------
#
# COMMENT OUT THIS PROCESS HERE AND ADD TO STEP 6 BECAUSE WE NEED
# TO CORRECT FOR ELEVATION FOR SATURATED DO.
# Loop over all rows
#for index, row in targets.iterrows():
#    #print('Temp '+str(row['Mean_Temp_Deg_C']))
#    #print('DO '+str(row['Mean_DO_mg_per_L']))
#    #print('DOsat '+str(row['Mean_DO_percent_saturation']))
#
#    # Must have temperature to attempt reconstruction
#    if ( not np.isnan(row['Mean_Temp_Deg_C']) ):
#        o2_sat_mg_per_l = o2sat.sw_o2sat(0.0, row['Mean_Temp_Deg_C'])*1.4291
#        #print('sw_O2_sat'+str(o2_sat_mg_per_l))
#
#        if (np.isnan(row['Mean_DO_mg_per_L']) and not np.isnan(row['Mean_DO_percent_saturation'])):
#            #print('Missing regular DO!')
#            # Compute any missing DO_mg_per_L from T and DOSAT  
#            targets.at[index,'Mean_DO_mg_per_L'] = row['Mean_DO_percent_saturation']*o2_sat_mg_per_l/100.0
#        elif (not np.isnan(row['Mean_DO_mg_per_L']) and np.isnan(row['Mean_DO_percent_saturation'])):
#            #print('Missing DOSAT')
#            # Compute any missing DOSAT from T and DO_mg_per_L.
#            targets.at[index,'Mean_DO_percent_saturation'] = 100.0*row['Mean_DO_mg_per_L']/o2_sat_mg_per_l 

# While Sample_Kit_ID will cover all the duplicates/replicates
# at the sites, there are three NEON sites that were time series
# sampled and as such need to be assigned the same group id even
# though they had different Sample_Kit_IDs.

# Convert Sample_Kit_ID to group IDs:
#---------------------------------
# Does not account for NEON sites
# with different Sample_Kit_ID
#targets['gid'] = targets['Sample_Kit_ID'].astype('category').cat.codes
#---------------------------------
# First make a list of all keys
# based on just Sample_Kit_ID.
group_key = targets["Sample_Kit_ID"].astype(str)

# Second, for each of the NEON sites with
# different Sample_Kit_IDs - there are only
# three - HOPB, MART, and MAYF - reassign
# them with the same site_code.
for site_code in ["HOPB", "MART", "MAYF"]:
    # Which sites have site_code in the Site_Name column?
    site_mask = targets["Site_Name"].str.contains(site_code, na=False)
    # For all of those detected sites, assign them the same site_code.
    group_key[site_mask] = site_code

# Find the unique group IDs based on the list of unique keys
targets["gid"] = group_key.astype("category").cat.codes

# There are two pairs of sites that need special merging:
# since they are within 50m of each other but do not share 
# Site_Name or a NEON marker.
merge_groups = [
    ["CM_001", "S19S_0055"],   # South Fork Palouse, Pullman WA
    ["CM_109", "S19S_0098"],   # Logan River, Logan UT
]

# Do the merge by assigning all gid that match
# the pairing the min gid of the two.
for kits in merge_groups:
    mask = targets["Sample_Kit_ID"].isin(kits)
    targets.loc[mask, "gid"] = targets.loc[mask, "gid"].min()

# Reassign gids that have been merged so that we
# don't have any breaks in the numbering.
targets["gid"] = targets["gid"].astype("category").cat.codes

# Drop Sample_Kit_ID
targets.drop(columns=['Sample_Kit_ID'], inplace=True)

# Drop Site_Name
# Site_Name was listed last in vars_to_use so it
# can be dropped with a pop() here.
vars_to_use.pop()
targets.drop(columns=['Site_Name'], inplace=True)

# Reorder columns:
vars_to_use[0] = 'Sample_ID'
vars_to_use[1] = 'gid'
targets = targets[vars_to_use]

# Save results
# Drop dataframe index
# Overwrite existing file
targets.to_csv('prep_01_output_train.csv', mode='w', index=False)

