#!/bin/bash
#================================
# Filter data to CONUS region
#================================

# Unzip a large file for temporary usage
gunzip -c us.xy.gz > us.xy.tmp

# Set the input/output
input=post_01_output_ml_predict_avg.csv
output=post_02_output_ml_pred_avg_filtered.csv

# First, get list of sites we want to filter out (already observed):
obsfile="../input_data/ICON_ModEx_Sites_To_Remove.csv"

# Remove the MP- prefix from the list, (but allow all SP- to remain!)
# (Matches Sample_ID format)
# Skip the header line
# Prefix all Sample_ID with "-e ^"<SAMPLE_ID>"," to force search at beginning of line
# and end with a comma to force avoiding confusing SP-5 with SP-50.
# Sort all and remove any diplicates
obslist=$(sed 's/MP-//g' $obsfile | awk ' NR > 1 {print "-e ^"$1","}' | sort | uniq )

# Need to add -e (-e's added above)
# Need to put Sample_ID in rightmost
# column so it is not interpreted as a lon,lat
# grep -v means invert selection  (i.e. remove the matches)
grep -v ${obslist} $input | awk -F, '{print $2,$3,$4,$5,$6,$7,$8,$9,$10,$1}' >> obs_filter.xyz.tmp

# Skip over 1 header line
gmt_cmd="gmt gmtselect obs_filter.xyz.tmp -Fus.xy.tmp -h1 > conus_filter.xyz.tmp"
sudo docker run --rm -v -v $(pwd):/work -w /work parallelworks/gmt $gmt_cmd
sudo chmod a+rw conus_filter.xyz.tmp

# Put Sample_ID in left column again, add commas, sort points by metric (last column)
awk '{OFS=","; print $10,$1,$2,$3,$4,$5,$6,$7,$8,$9}' conus_filter.xyz.tmp | sort -n +10 > $output

# Clean up
rm -f obs_filter.xyz.tmp
rm -f conus_filter.xyz.tmp
rm -f us.xy.tmp

