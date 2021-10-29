#!/bin/bash

URL="https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=%s&referred_module=sw&period=&begin_date=2010-01-01&end_date=2021-10-27"

IDS=$(grep -Eo  "  \* [0-9]{8}" ./acoe_data/meta_data/usgs_sg_stat.md | awk '{print $2}')

for id in ${IDS[@]}; do
    url=$(printf $URL $id)
    file=$(printf "%s.tsv" $id)
    curl $url --output $file --silent &
done

