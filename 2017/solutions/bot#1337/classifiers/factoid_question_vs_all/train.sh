#!/usr/bin/env bash

../fastText/fasttext supervised -input train.txt -output model -epoch 50 -wordNgrams 2
../fastText/fasttext test model.bin test.txt
