# Coordinate data intake

import pandas as pd
import numpy as np
import o2sat

data = pd.read_csv("../input_data/ICON_ModEx_Collaborator_Site_Water_Chemistry_Data.csv")

# List the variables we want here.
# Order of the list sets left-to-right column order in dataframe/csv output
# Sample_Kit_ID is not unique - shared among sites
# Put oxygen last b/c need to add DOSAT on the left side.
vars_to_use=[
    'Site_ID',
    'Sample_Latitude',
    'Sample_Longitude',
    'Mean_Temp_Deg_C',
    'pH',
    'Mean_DO_mg_per_L'
]

# Grab a view of just the subset we want
core_vars = data[vars_to_use]

# Add a column for DOSAT, all NaN
core_vars.insert(
    len(core_vars.columns),
    'Mean_DO_percent_saturation',
    np.nan)

# Change all -9999 values to NaN
core_vars.replace(
    to_replace=-9999,
    value=np.nan,
    inplace=True)

# NO RESPIRATION RATES IN PREDICT DATA
# DO NOT FILTER MISSING VALUES - REPLACE
# WITH MEANS LATER

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

# Loop over all rows
for index, row in core_vars.iterrows():
    #print('Temp '+str(row['Mean_Temp_Deg_C']))
    #print('DO '+str(row['Mean_DO_mg_per_L']))
    #print('DOsat '+str(row['Mean_DO_percent_saturation']))

    # Must have temperature to attempt reconstruction
    if ( not np.isnan(row['Mean_Temp_Deg_C']) ):
        o2_sat_mg_per_l = o2sat.sw_o2sat(0.0, row['Mean_Temp_Deg_C'])*1.4291
        #print('sw_O2_sat'+str(o2_sat_mg_per_l))

        if (np.isnan(row['Mean_DO_mg_per_L']) and not np.isnan(row['Mean_DO_percent_saturation'])):
            #print('Missing regular DO!')
            # Compute any missing DO_mg_per_L from T and DOSAT  
            core_vars.at[index,'Mean_DO_mg_per_L'] = row['Mean_DO_percent_saturation']*o2_sat_mg_per_l/100.0
        elif (not np.isnan(row['Mean_DO_mg_per_L']) and np.isnan(row['Mean_DO_percent_saturation'])):
            #print('Missing DOSAT')
            # Compute any missing DOSAT from T and DO_mg_per_L.
            core_vars.at[index,'Mean_DO_percent_saturation'] = 100.0*row['Mean_DO_mg_per_L']/o2_sat_mg_per_l 

# Save results
core_vars.to_csv('prep_02_output.csv', mode='w')
