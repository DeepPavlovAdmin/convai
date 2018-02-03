#!/usr/bin/env bash

###
# RUN SHORT TERM RANKER IN EXPLORATION MODE
###
# hyperdash run -n "explore short term ranker" python train.py \
#     data/voted_data_db_1510012489.57.json \
#     data/voted_data_round1_1510012503.6.json \
#     short_term \
#     --gpu 1 \
#     --explore 1000 \
#     --threshold 0.635

###
# RUN LONG TERM RANKER IN EXPLORATION MODE
###
hyperdash run -n "explore long term ranker" python train.py \
    ./data/full_data_db_1510012482.99.json \
    ./data/full_data_round1_1510012496.02.json \
    long_term \
    --gpu 0 \
    --explore 1000 \
    --threshold -1.56

