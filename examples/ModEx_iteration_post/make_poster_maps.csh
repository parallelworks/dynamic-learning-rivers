#!/bin/tcsh -f
#====================

# Set input file
set infile = sl_pca_consolidated.csv

# Grab just the lon, lat, avg metric, slope, and r2
awk -F, '{print $3,$4,$63,$65,$67}' $infile > tmp.xyasr

# Set output files
set outps1 = map_max_min_metric.ps
set outps2 = map_max_min_slopes.ps

# Find mean and std from the data set with:
set mean = `gmt gmtmath tmp.xyasr -C2 MEAN -Sl =`
set std = `gmt gmtmath tmp.xyasr -C2 STD 1.0 MUL -Sl =`

set high = `gmt gmtmath -Q $mean[3] $std[3] ADD = `
set low = `gmt gmtmath -Q $mean[3] $std[3] SUB = `

echo $high
echo $low

# Filter the data
awk -v val=$high 'NR > 1 && $3 > val {print $1,$2}' tmp.xyasr > high.xy.tmp
awk -v val=$low 'NR > 1 && $3 < val {print $1,$2}' tmp.xyasr > low.xy.tmp

wc -l high.xy.tmp
wc -l low.xy.tmp

gmt psbasemap -JM6i -R-125/-65/22/50 -Ba15/a5 -P -K -X1i -Y2i > $outps1
gmt pscoast -J -R -B -P -O -K -Gdarkgray >> $outps1
#gmt psxy low.xy.tmp -J -R -B -P -O -K -Sp0.1 -Wblack -Gblack >> $outps1
gmt psxy high.xy.tmp -J -R -B -P -O -K -S+0.2 -Wblack -Gblack >> $outps1

# Find the regions of maxium variability in slope
set high = `gmt gmtmath -Q $mean[5] $std[5] ADD = `
awk -v val=$high 'NR > 1 && $5 > val {print $1,$2}' tmp.xyasr > high.xy.tmp
gmt psxy high.xy.tmp -J -R -B -P -O -K -Sc0.2 -Wblack >> $outps1

# Find the regions of maxium slope
set high = `gmt gmtmath -Q $mean[4] $std[4] ADD = `
awk -v val=$high 'NR > 1 && $4 > val {print $1,$2}' tmp.xyasr > high.xy.tmp
gmt psxy high.xy.tmp -J -R -B -P -O -K -Sp0.1 -Wred -Gred >> $outps1

# Clean up
ps2pdf $outps1
rm -f $outps1
rm -f low.xy.tmp
rm -f high.xy.tmp

#=======================================
# Plot errors

# Get errors
#awk '{print $1,$2,$4}' tmp.xyr > tmp.xye

#gmt makecpt -T0/5/1 -Cno_green > tmp.cpt

#gmt psbasemap -JM6i -R-125/-65/22/50 -Ba15/a5 -P -K -Y2i > $outps2
#gmt pscoast -J -R -B -P -O -K -Gdarkgray >> $outps2
#gmt psxy tmp.xye -J -R -B -Ctmp -P -O -K -Sp0.1 >> $outps2
#gmt psscale -Dx0i/-0.75i+w6i/0.25i+e+h -Ctmp -Ba5g1 -B+l"Respiration\ rate error estimate,\ mg\ O2\/L\/h" -P -O -K >> $outps2

#gmt psbasemap -J -R -Ba15/a5 -P -O -K -Y5i >> $outps2
#gmt pscoast -J -R -B -P -O -K -Gdarkgray >> $outps2

# Find mean and std from the data set with:
#set mean = `gmt gmtmath tmp.xye -C2 MEAN -Sl =`
#set std = `gmt gmtmath tmp.xye -C2 STD 1.0 MUL -Sl =`

#echo "Error estimate statistics..."
#echo "Mean: "$mean

#set high = `gmt gmtmath -Q $mean[3] $std[3] ADD = `
#set low = `gmt gmtmath -Q $mean[3] $std[3] SUB = `

#echo "High: "$high
#echo "Low:  "$low

# Filter the data
#awk -v val=$high 'NR > 1 && $3 > val {print $1,$2}' tmp.xye > high.xy.tmp
#awk -v val=$low 'NR > 1 && $3 < val {print $1,$2}' tmp.xye > low.xy.tmp

#wc -l high.xy.tmp
#wc -l low.xy.tmp

#gmt psxy low.xy.tmp -J -R -B -P -O -K -Sp0.1 -Wblack -Gblack >> $outps2
#gmt psxy high.xy.tmp -J -R -B -P -O -K -Sp0.1 -Wred -Gred >> $outps2

# Final Clean up
rm -f tmp.xyasr
#rm -f tmp.cpt
#rm -f high.xy.tmp
#rm -f low.xy.tmp
#rm -f tmp.xye
#ps2pdf $outps2
#rm -f $outps2
