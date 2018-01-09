#!/usr/bin/env bash

../fastText/fasttext supervised -input train_balance_shuf_tokenized.txt -output model -epoch 25 -wordNgrams 2 -loss hs -lr 1.0 -bucket 200000 -dim 50
../fastText/fasttext test model.bin tests_balance_tokenized.txt
