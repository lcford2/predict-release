#!/bin/bash

YEARS=$(seq 1990 2020)
MONTHS=$(seq -f "%02g" 1 12)

for y in $YEARS; do
    for m in $MONTHS; do
        URL=$(printf "https://www.usbr.gov/lc/region/g4000/cy%d/%s_%d.out" $y $m $y)
        FILE=$(printf "./raw_data/%s_%d.out" $m $y)
        curl $URL --output $FILE --silent
    done    
done
