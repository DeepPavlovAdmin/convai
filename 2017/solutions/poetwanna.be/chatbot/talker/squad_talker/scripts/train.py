from __future__ import print_function

import numpy as np
import os
import sys
import argparse

'''
    Training script for QANet. Make sure you are using GPU for Theano
    computations.
    Libraries originally used: Theano 0.9, cuda 8.0, cudnn 5.1, lasagne 0.2

        --glove             Path to glove.6B.300d.txt GloVe embeddings
        --squad             Path to SQuAD data set. A directory containing
                                dev-v1.1.json, train-v1.1.json, and a folder
                                'preproc' with preprocessed data (see
                                prep_squad.py). Later referred to as
                                <squad_path>.
        --output-dir        Output directory, default is 'output'
        --save_preds        Whether or not to save intermediate predictions
                                on dev-v1.1. Binary flag.
        --batch_size        Default is 30
        --learning_rate     Default is 0.001 (ADAM)
        --checkpoint        How many samples between checkpoints, default 64000
        --negative          A list of negative data sets, for negative answers
                                experiment. Each entry is a name of a directory
                                in <squad_path>/preproc. File structure for
                                'wiki_neg' data would be:

                                <squad_path>/
                                    dev-v1.1.json         <-- raw squad data
                                    train-v1.1.json       <--

                                    preproc/
                                        dev.json           <-- preprocced squad
                                        dev_bin_feats.json  <--
                                        dev_char_ascii.json  <--
                                        dev_words.json       <--
                                        train.json           <--
                                        train_bin_feats.json  <--
                                        train_char_ascii.json  <--
                                        train_words.json       <--

                                        wiki_neg/        <-- negative data set
                                            train_bin_feats.json
                                            train_char_ascii.json
                                            train_words.json

                                Negative data sets must be downloaded
                                separately (alongside the rest of chatbot data)
                                and preprocessed with prep_squad_neg.py.

                                Example: --negative wiki_pos squad_neg_rng

    Best model is saved as <output_dir>/6B.best.npz (or 6B.best.neg.npz in case
    of negative training). You need to manualy copy it to the chatbot's
    squad_models data directory to use it in a bot. Additionaly, to enable
    negative answers in a bot, set the squad_negative flag in chatbot/config.py
'''

parser = argparse.ArgumentParser(description='Train script for QANet.')
parser.add_argument('-g', '--glove', default='data/glove.6B.300d.txt')
parser.add_argument('-s', '--squad', default='data/squad/')
parser.add_argument('-o', '--output_dir', default='output')
parser.add_argument('--save_preds', action='store_true')
parser.add_argument('-bs', '--batch_size', default=64, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('-cp', '--checkpoint', default=64000, type=int)
parser.add_argument('-n', '--negative', nargs='+', default=[])

args = parser.parse_args()

# set paths
squad_prep_path = os.path.join(args.squad, 'preproc')

preds_path = os.path.join(args.output_dir, 'pred') if args.save_preds else None

if not os.path.exists(args.glove):
    sys.exit("GloVe embeddings not found at " + args.glove + ". Aborting.")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
elif os.listdir(args.output_dir):
    sys.exit(
        "Chosen output directory already exists and is not empty. Aborting.")
if preds_path is not None and not os.path.exists(preds_path):
    os.makedirs(preds_path)

# redirect all prints to log file
log_path = os.path.join(args.output_dir, 'log')
print("All prints are redirected to", log_path)
log = open(log_path, 'w', buffering=1)
sys.stderr = log
sys.stdout = log

sys.path.append('../')
from QANet import QANet
from squad_tools import load_squad_train, load_squad_dev, \
    filter_empty_answers, trim_data, train_QANet, load_glove

print("\nRun params:")
for arg in vars(args):
    print(arg.ljust(25), getattr(args, arg))
print("\n")

################################

print("Loading data...")

glove_words, glove_embs = load_glove(args.glove)
voc_size = glove_embs.shape[0]

NAW_token = glove_words.index('<not_a_word>')

train_data = load_squad_train(squad_prep_path, negative_paths=args.negative,
                              NAW_token=NAW_token)
train_data = filter_empty_answers(train_data)
train_data = trim_data(train_data, 300)

dev_data = load_squad_dev(args.squad, squad_prep_path, NAW_token=NAW_token,
                          lower_raw=True,
                          make_negative=bool(args.negative))

net = QANet(voc_size=voc_size,
            emb_init=glove_embs,
            dev_data=dev_data,
            predictions_path=preds_path,
            train_unk=True,
            negative=bool(args.negative),
            init_lrate=args.learning_rate,
            checkpoint_examples=args.checkpoint,
            conv='valid')

train_QANet(net, train_data, args.output_dir, batch_size=args.batch_size)
