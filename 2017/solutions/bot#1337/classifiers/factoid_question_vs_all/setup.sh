#!/usr/bin/env bash

gshuf -n 300000 comments_with_label.txt > comments_300k.txt
cat ../squad/questions.txt ../cornell\ movie-dialogs\ corpus/only_lines.txt comments_300k.txt squad_answers_with_label.txt > data.txt
gshuf data.txt > data_shuf.txt
python my-tokenize.py data_shuf.txt > data_shuf_tokenized.txt
head -n 81463 data_shuf_tokenized.txt > test.txt
tail -n 733174 data_shuf_tokenized.txt > train.txt
