#!/bin/bash

for depth in {1..10};
do
    python tclr_model.py all -d $depth
done
