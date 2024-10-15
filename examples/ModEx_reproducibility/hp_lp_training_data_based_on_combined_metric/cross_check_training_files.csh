#!/bin/csh
#============================
# Check all Sample ID are the same

foreach file ( `ls ?p_???.csv` )
echo $file
awk -F, '{print $1}' $file | sort > csv.tmp
awk -F, '{print $2}' ${file}.out | sort > csv.out.tmp
diff csv.tmp csv.out.tmp
end


