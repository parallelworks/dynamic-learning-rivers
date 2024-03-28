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
# Sample_Kit_ID is not unique - shared among sites
vars_to_use=[
    'Sample_ID',
    'Sample_Longitude',
    'Sample_Latitude',
    'Mean_Temp_Deg_C',
    'pH',
    'Mean_DO_mg_per_L',
    'Mean_DO_percent_saturation',
    target_name
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

# Save results
# Drop dataframe index
# Overwrite existing file
targets.to_csv('prep_01_output_train.csv', mode='w', index=False)
