#!/bin/bash

for depth in {1..5};
do
    python tclr_model.py all -d $depth
done
