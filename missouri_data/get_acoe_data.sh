#!/bin/bash

#LOC_IDS=(3161030 2998030 3091030 3155030 3097030)
#LOCS=("garrison" "fort_peck" "fort_randall" "big_bend" "gavins_point")
LOC_IDS=(4008030)
LOCS=("oahe")

URL="https://water.usace.army.mil/a2w/CWMS_CRREL.cwms_data_api.get_report_json?p_location_id=%s&p_parameter_type=Flow:Stor:Precip:Stage:Elev&p_last=10&p_last_unit=years&p_unit_system=EN&p_format=CSV"

for i in $(seq 0 4); do
    loc_id=${LOC_IDS[$i]}
    loc=${LOCS[$i]}
    url=$(printf $URL $loc_id)
    echo $url
    file=$(printf "./acoe_data/csv/%s.csv" $loc)
    curl $url --output $file --silent &
done
