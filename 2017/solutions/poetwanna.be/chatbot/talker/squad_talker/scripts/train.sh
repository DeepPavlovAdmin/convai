#!/bin/bash
#
# Training the QA model.
# Arguments:
#   1  path glove.6B.300d.txt GloVe embeddings
#           from http://nlp.stanford.edu/data/glove.6B.zip
#   2  path to a directory containing dev-v1.1.json and train-v1.1.json
#           from https://rajpurkar.github.io/SQuAD-explorer/
#
# Edit BATCH_SIZE in case of memory problems.
#
# Best model is saved as output/6B.best.npz (or 6B.best.neg.npz in case
# of negative training). You need to manualy copy it to the chatbot's
# squad_models data directory to use it in a bot.
#
#
# For replicating the negative answers experiment from the ConvAI submission
# paper, use prep_squad_neg.py, train.py, and test_neg.py directly.
# See respective files for details.

BATCH_SIZE="64"

python -u prep_squad.py --glove=$1 --squad=$2
python -u train.py --glove=$1 --squad=$2 --batch_size=$BATCH_SIZE
