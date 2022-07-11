#!/bin/bash

# grid search
# for depth in 1 2 3 4 5 6 7 8
# do
#     for mss in '0.0' '0.01' '0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.1' '0.15' '0.20';
#     do
#         python tclr_model.py all -d $depth --mss $mss
#     done
# done

# assim search
# td 5 mss 0.01
# td 2 mss 0.2
for assim in 'daily' 'weekly' 'monthly' 'seasonally' 'semi-annually' 'annually'
do
    python tclr_model.py all -d 5 --mss 0.01 --assim $assim
    python tclr_model.py all -d 2 --mss 0.2 --assim $assim
done