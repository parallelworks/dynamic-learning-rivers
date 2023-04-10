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
# Skip over header line.
# Remove all "SP-" prefixes because they are converted to NaN by GMT later
grep -v ${obslist} $input | awk -F, 'NR > 1 {print $2,$3,$4,$5,$6,$7,$8,$9,$10,$1}' | sed 's/SP-//g' > obs_filter.xyz.tmp

# Get header for later
awk 'NR == 1 {print $0}' $input > $output

# Check Docker daemon is running
if [ `sudo systemctl is-active docker` == "active" ]
then
    #echo Docker daemon is already started. Do nothing.
    sleep 1
else
    #echo Docker daemon not started. Starting Docker daemon...
    sudo systemctl start docker
fi

# Select points only inside CONUS polygon
# Docs say -a9=name:STRING should do the trick, but doesn't work for me.
# (https://docs.generic-mapping-tools.org/6.0/gmt.html#aspatial-full)
gmt_cmd="gmt gmtselect obs_filter.xyz.tmp -Fus.xy.tmp > conus_filter.xyz.tmp"
sudo docker run --rm -v $(pwd):/work -w /work parallelworks/gmt $gmt_cmd
sudo chmod a+rw conus_filter.xyz.tmp

# Put Sample_ID in left column again, add commas, sort points by metric (last column)
# Insert "SP-" again for sampler picked sites and "MP-" for ML predicted sites.
awk '{OFS=","; if ($10 < 10000 ) {print "SP-"$10,$1,$2,$3,$4,$5,$6,$7,$8,$9} else {print "MP-"$10,$1,$2,$3,$4,$5,$6,$7,$8,$9} }' conus_filter.xyz.tmp | sort -t "," -n +9 -10 >> $output

# Clean up
rm -f obs_filter.xyz.tmp
rm -f conus_filter.xyz.tmp
rm -f us.xy.tmp
