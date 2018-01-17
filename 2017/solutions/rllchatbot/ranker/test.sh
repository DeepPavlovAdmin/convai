#!/usr/bin/env bash

###
# DEBUG and TEST script to get accuracy of a model on test set
###
python test.py \
    ./models/short_term/0.641391/1510248853.21_Estimator_ \
    ./models/long_term/1.4506/1510248853.21_short_term.0641391.151024885321_Estimator__ \
    --gpu 3

