from __future__ import print_function

import json
import os
import argparse

from os.path import join
from squad_tools import load_glove
from my_tokenize import tokenize

'''
    Negative SQuAD data preprocessor for QANet. Run this before training with
    negatives, after prep_squad.py.

        --glove     Path to glove.6B.300d.txt GloVe embeddings
        --squad     SQuAD path, should be the same as in prep_squad.py
                        Requires the following structure:

                        <squad_path>/
                            preproc/
                                train.json
                                train_bin_feats.json
                                train_char_ascii.json
                                train_words.json
                                ...

                            negative_samples/
                                train.wiki.pos.json
                                train.wiki.neg.json
                                train.squad.random.json
                                ...

    negative_samples are contained in the chatbot data pack in squad/ directory

    Result is saved to sepatate folders in <squad_path>/preproc/
'''

parser = argparse.ArgumentParser(
    description='Negetive data preprocessor for QANet.')
parser.add_argument('-g', '--glove', default='data/glove.6B.300d.txt')
parser.add_argument('-s', '--squad', default='data/squad/')
args = parser.parse_args()

wordlist = load_glove(args.glove, only_words=True)
w_to_i = {w: i for i, w in enumerate(wordlist)}

with open(join(args.squad, 'preproc/train.json')) as f:
    train = json.load(f)
with open(join(args.squad, 'preproc/train_words.json')) as f:
    train_words = json.load(f)
with open(join(args.squad, 'preproc/train_char_ascii.json')) as f:
    train_char_ascii = json.load(f)
with open(join(args.squad, 'preproc/train_bin_feats.json')) as f:
    train_bin_feats = json.load(f)


NAW_tok = u'<not_a_word>'
NAW_idx = wordlist.index(NAW_tok)
NAW_chars = [1, 3, 2]


def make_bin_feats(sample):
    q, x = sample
    qset = set(q)
    return [w in qset for w in x]


def get_word_nums(s):
    return [w_to_i.get(w, 0) for w in s]


chars = [unichr(i) for i in range(128)]
c_to_i = {chars[i]: i for i in range(128)}


def get_char_nums_for_word(w):
    return [1] + [c_to_i.get(c, 0) for c in w] + [2]


def cut_sentence_with_ans(ans, context):
    i, j = min(ans), max(ans)
    while i > 0 and context[i-1] != '.':
        i -= 1
    while context[j] != '.' and j < len(context) - 1:
        j += 1
    return i, j + 1


# squad_neg_cut


print("Preparing squad-cut data...")

out_path = join(args.squad, 'preproc/squad_neg_cut')

train_new = []
train_words_new = []
train_char_new = []
train_bin_feats_new = []

for k in range(len(train)):
    if not train_words[k][0]:
        continue

    ans = train_words[k][0][0]
    i, j = cut_sentence_with_ans(ans, train[k][2])

    new_raw = train[k][2][:i] + train[k][2][j:] + [NAW_tok]
    train_new.append([[], train[k][1], new_raw])

    new_words = train_words[k][2][:i] + train_words[k][2][j:] + [NAW_idx]
    new_ans = [[len(new_words) - 1]]
    train_words_new.append([new_ans, train_words[k][1], new_words])

    new_char = train_char_ascii[k][1][:i] + train_char_ascii[k][1][j:] + [NAW_chars]
    train_char_new.append([train_char_ascii[k][0], new_char])

    new_bin_feats = train_bin_feats[k][:i] + train_bin_feats[k][j:] + [False]
    train_bin_feats_new.append(new_bin_feats)

if not os.path.exists(out_path):
    os.makedirs(out_path)

print("Saving files to", out_path)

with open(join(out_path, 'train_words.json'), 'w') as f:
    json.dump(train_words_new, f)
with open(join(out_path, 'train_char_ascii.json'), 'w') as f:
    json.dump(train_char_new, f)
with open(join(out_path, 'train_bin_feats.json'), 'w') as f:
    json.dump(train_bin_feats_new, f)


# squad_neg_rng


print("Preparing squad-rng data...")

out_path = join(args.squad, 'preproc/squad_neg_rng')

with open(join(args.squad, 'negative_samples/train.squad.random.json')) as f:
    squad_random_order = json.load(f)

random_train_words = []
random_train_bin_feats = []
random_train_char_ascii = []

for i in range(len(train_words)):
    par_nr = squad_random_order[i]
    q = train_words[i][1]
    x = train_words[par_nr][2] + [NAW_idx]
    a = [[len(x) - 1]]
    random_train_words.append([a, q, x])

    q_toks = train[i][1]
    x_toks = train[par_nr][2]
    random_train_bin_feats.append(make_bin_feats([q_toks, x_toks]) + [False])

    q_char = train_char_ascii[i][0]
    x_char = train_char_ascii[par_nr][1] + [[1, 3, 2]]
    random_train_char_ascii.append([q_char, x_char])

if not os.path.exists(out_path):
    os.makedirs(out_path)

print("Saving files to", out_path)

with open(join(out_path, 'train_words.json'), 'w') as f:
    json.dump(random_train_words, f)
with open(join(out_path, 'train_bin_feats.json'), 'w') as f:
    json.dump(random_train_bin_feats, f)
with open(join(out_path, 'train_char_ascii.json'), 'w') as f:
    json.dump(random_train_char_ascii, f)


# wiki_neg


print("Preparing wiki-neg data...")

out_path = join(args.squad, 'preproc/wiki_neg')

with open(join(args.squad, 'negative_samples/train.wiki.neg.json')) as f:
    wiki_data_neg = json.load(f)

for d in wiki_data_neg:
    d[1] = tokenize(d[1].lower())
    d[2] = tokenize(d[2].lower())
    d[0] = [[len(d[2])]]
    d[2].append(NAW_tok)

wiki_neg_words = []
for a, q, x in wiki_data_neg:
    q_num = get_word_nums(q)
    x_num = get_word_nums(x)
    wiki_neg_words.append([a, q_num, x_num])

wiki_neg_bin_feats = []
for _, q, c in wiki_data_neg:
    wiki_neg_bin_feats.append(make_bin_feats([q, c]))

wiki_neg_ascii_chars = []
for _, q, x in wiki_data_neg:
    q_char = map(get_char_nums_for_word, q)
    x_char = map(get_char_nums_for_word, x[:-1]) + [[1, 3, 2]]
    wiki_neg_ascii_chars.append([q_char, x_char])

if not os.path.exists(out_path):
    os.makedirs(out_path)

print("Saving files to", out_path)

with open(join(out_path, 'train_words.json'), 'w') as f:
    json.dump(wiki_neg_words, f)
with open(join(out_path, 'train_bin_feats.json'), 'w') as f:
    json.dump(wiki_neg_bin_feats, f)
with open(join(out_path, 'train_char_ascii.json'), 'w') as f:
    json.dump(wiki_neg_ascii_chars, f)


# wiki_pos


print("Preparing wiki-pos data...")

out_path = join(args.squad, 'preproc/wiki_pos')

with open(join(args.squad, 'negative_samples/train.wiki.pos.json')) as f:
    wiki_data_pos = json.load(f)

wiki_data_pos_prep = []
for d in wiki_data_pos:
    q = tokenize(d[1].lower())
    x = tokenize(d[3].lower())
    ans = [list(range(d[4], d[4] + len(tokenize(d[2]))))]
    x.append(NAW_tok)
    wiki_data_pos_prep.append([ans, q, x])

wiki_pos_words = []
for a, q, x in wiki_data_pos_prep:
    q_num = get_word_nums(q)
    x_num = get_word_nums(x)
    wiki_pos_words.append([a, q_num, x_num])

wiki_pos_bin_feats = []
for _, q, c in wiki_data_pos_prep:
    wiki_pos_bin_feats.append(make_bin_feats([q, c]))

wiki_pos_ascii_chars = []
for _, q, x in wiki_data_pos_prep:
    q_char = map(get_char_nums_for_word, q)
    x_char = map(get_char_nums_for_word, x[:-1]) + [[1, 3, 2]]
    wiki_pos_ascii_chars.append([q_char, x_char])

if not os.path.exists(out_path):
    os.makedirs(out_path)

print("Saving files to", out_path)

with open(join(out_path, 'train_words.json'), 'w') as f:
    json.dump(wiki_pos_words, f)
with open(join(out_path, 'train_bin_feats.json'), 'w') as f:
    json.dump(wiki_pos_bin_feats, f)
with open(join(out_path, 'train_char_ascii.json'), 'w') as f:
    json.dump(wiki_pos_ascii_chars, f)
