#!/bin/bash

# ~/Downloads/bigartm-0.8.3/build/bin/bigartm --read-vw-corpus vw.wiki-en.txt \
#   --topics "topic:15,noise:3" --batch-size 1000 --save-model wiki15+3topics.init.model \
#   --write-predictions wiki15+3topics.theta.txt --write-model-readable wiki15+3topics.phi.txt \
#   --num-collection-passes 5 \
#   --update-every 5 \
#   --save-batches wiki15+3topics.batches \
#   --save-dictionary wiki15+3topics.dict \
#   --threads 8 --force


~/Downloads/bigartm-0.8.3/build/bin/bigartm \
    --topics "topic:15,noise:3"  \
    --use-dictionary wiki15+3topics.dict \
    --use-batches wiki15+3topics.batches \
    --load-model wiki15+3topics.init.model \
    --regularizer "0.5 SmoothPhi #noise" \
    --regularizer "0.5 SparsePhi #topic" \
    --regularizer "0.5 SmoothTheta #noise" \
    --regularizer "0.5 SparseTheta #topic" \
    --regularizer "1000 Decorrelation #topic" \
    --num-collection-passes 10 \
    --update-every 3 \
    --tau0 128 \
    --kappa 0.5 \
    --save-model wiki15+3topics.new.model \
    --write-predictions wiki15+3topics.new.theta.txt \
    --write-model-readable wiki15+3topics.new.phi.txt \
    --threads 8
