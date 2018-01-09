#!/bin/bash

DATA_FILENAME="$1"
WRITE_PRED_FILENAME="$2"

bigartm \
--topics "topic:15,noise:3"  \
--use-dictionary wiki15+3topics.dict \
--load-model wiki15+3topics.new.model \
--regularizer "0.5 SmoothPhi #noise" \
--regularizer "0.5 SparsePhi #topic" \
--regularizer "0.5 SmoothTheta #noise" \
--regularizer "0.5 SparseTheta #topic" \
--regularizer "1000 Decorrelation #topic" \
--tau0 128 \
--kappa 0.5 \
--write-predictions "$WRITE_PRED_FILENAME" \
--read-vw-corpus "$DATA_FILENAME" \
--force
