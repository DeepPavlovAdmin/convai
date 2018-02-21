from __future__ import print_function

import json
import argparse
import os
import sys
import numpy as np

'''
    Negative testing for QANet. Shows how many negative answer the network
    gives on four small testing data sets. Run this after training a model
    for negative answers.

        --glove             Path to glove.6B.300d.txt GloVe embeddings
        --batch_size        Default is 10
        --model             Path to a .npz trained model file.
        --squad             <squad_path>. The following directory structure
                                is expected:

                                <squad_path>/
                                    preproc/
                                        [preprocced data including dev.json]
                                    negative_samples/
                                        dev.wiki.pos.json
                                        dev.wiki.neg.json
                                        dev.squad.random.json
                                        ...

    The negative_samples/ data must be downloaded separately.
'''


parser = argparse.ArgumentParser(description='Negative testing for QANet.')
parser.add_argument('-g', '--glove', default='data/glove.6B.300d.txt')
parser.add_argument('-s', '--squad', default='data/squad/')
parser.add_argument('-bs', '--batch_size', default=10, type=int)
parser.add_argument('-m', '--model')

args = parser.parse_args()

output_dir = os.path.dirname(args.model)

log_path = os.path.join(output_dir, 'neg_test')
print("All prints are redirected to", log_path)
log = open(log_path, 'w', buffering=1)
sys.stderr = log
sys.stdout = log

from AnswerBot_test import AnswerBot
from AnswerBot_test import not_a_word_Str as NAW_tok
from squad_tools import load_glove

print("\nRun params:")
for arg in vars(args):
    print(arg.ljust(25), getattr(args, arg))
print("\n")


def iterate_minibatches(data):
    for start_idx in range(0, len(data), args.batch_size):
        excerpt = slice(start_idx, start_idx + args.batch_size)
        yield zip(*data[excerpt])


def calc_naws(data):
    naws = 0
    for batch in iterate_minibatches(data):
        qs, xs = batch
        ans = [a[0] for a in abot.get_answers(qs, xs)]
        naws += len([a for a in ans if NAW_tok in a])
    return naws

# Build AnswerBot
glove_dict, glove_embs = load_glove(args.glove)

abot = AnswerBot(args.model, glove_embs, glove_dict, '6B',
                 train_unk=True,
                 negative=True,
                 conv='valid')

print('\nDiscarded paragraphs:\n')

# wiki_pos
path_wiki_pos = os.path.join(
    args.squad, 'negative_samples', 'dev.wiki.pos.json')
with open(path_wiki_pos) as f:
    wiki_pos = [[d[1], d[3]] for d in json.load(f)]
naws_wiki_pos = calc_naws(wiki_pos)
print('wiki pos: %.2f' % (float(naws_wiki_pos) / len(wiki_pos)))

# wiki_neg
path_wiki_neg = os.path.join(
    args.squad, 'negative_samples', 'dev.wiki.neg.json')
with open(path_wiki_neg) as f:
    wiki_neg = [d[1:] for d in json.load(f)]
naws_wiki_neg = calc_naws(wiki_neg)
print('wiki neg: %.2f' % (float(naws_wiki_neg) / len(wiki_neg)))

# regular
path_dev = os.path.join(args.squad, 'preproc', 'dev.json')
with open(path_dev) as f:
    dev = [d[1:3] for d in json.load(f)]
naws_dev = calc_naws(dev)
print('dev     : %.2f' % (float(naws_dev) / len(dev)))

# randomized
path_dev_rng = os.path.join(
    args.squad, 'negative_samples', 'dev.squad.random.json')
with open(path_dev_rng) as f:
    dev_rng = json.load(f)
naws_dev_rng = calc_naws(dev_rng)
print('dev rng : %.2f' % (float(naws_dev_rng) / len(dev_rng)))
