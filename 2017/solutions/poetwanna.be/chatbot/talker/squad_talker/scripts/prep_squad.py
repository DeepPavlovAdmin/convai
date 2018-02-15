from __future__ import print_function

import json
import os
import argparse
from os.path import join

from squad_tools import load_glove
from my_tokenize import tokenize_with_ans_idx as tokenize

'''
    SQuAD data preprocessor for QANet. Run this before training.

        --glove     Path to glove.6B.300d.txt GloVe embeddings.
                        This is needed to gather the vocabulary.
        --squad     Path to SQuAD data set. A directory containing
                        dev-v1.1.json and train-v1.1.json

    Result is saved to <squad_path>/preproc/
'''

parser = argparse.ArgumentParser(description='Data preprocessor for QANet.')
parser.add_argument('-g', '--glove', default='data/glove.6B.300d.txt')
parser.add_argument('-s', '--squad', default='data/squad/')

args = parser.parse_args()


out_path = join(args.squad, 'preproc')

if not os.path.exists(out_path):
    os.makedirs(out_path)


with open(join(args.squad, 'dev-v1.1.json')) as f:
    dev = json.load(f)
with open(join(args.squad, 'train-v1.1.json')) as f:
    train = json.load(f)

wordlist = load_glove(args.glove, only_words=True)
w_to_i = {w: i for i, w in enumerate(wordlist)}

print("Preparing data. This may take a while.")

######################

def prepare_data(json_data, withId=False):
    data = []

    for par in json_data['data']:
        for con in par['paragraphs']:
            context = con['context'].lower()

            for q in con['qas']:
                question = q['question'].lower()
                question_tok = tokenize(question)[0]
                answers = []

                Id = q['id']

                for ans in q['answers']:
                    text = ans['text'].lower()
                    ans_start = ans['answer_start']
                    context_tok, ans_start = tokenize(context, ans_start)
                    ans_end = ans_start + len(tokenize(text)[0]) - 1

                    answers.append(([ans_start, ans_end], text))

                data.append([answers, question_tok, context_tok])
                if withId:
                    data[-1].append(Id)
    return data


data_train = prepare_data(train)
data_dev = prepare_data(dev, withId=True)


# throw out the questions with answers we can't possibly learn
# (the ones that aren't whole words)
data_train = [d for d in data_train if
              u' '.join(d[2][d[0][0][0][0]:d[0][0][0][1] + 1]) ==
              u' '.join(tokenize(d[0][0][1])[0])]


def get_word_nums(s):
    return [w_to_i.get(w, 0) for w in s]


train_num = []
for a, q, x in data_train:
    a_num = list(range(a[0][0][0], a[0][0][1] + 1))
    q_num = get_word_nums(q)
    x_num = get_word_nums(x)
    train_num.append([[a_num], q_num, x_num])

dev_num = []
for a, q, x, _ in data_dev:
    q_num = get_word_nums(q)
    x_num = get_word_nums(x)
    dev_num.append([[], q_num, x_num])


def make_bin_feats(sample):
    q, x = sample[1:3]
    qset = set(q)
    return [w in qset for w in x]


train_bin_feats = map(make_bin_feats, data_train)
dev_bin_feats = map(make_bin_feats, data_dev)


# for character embeddings
# 0 - unk
# 1 - start
# 2 - end
chars = [unichr(i) for i in range(128)]
c_to_i = {chars[i]: i for i in range(128)}


def get_char_nums_for_word(w):
    return [1] + [c_to_i.get(c, 0) for c in w] + [2]


def prepare_chars(data):
    data_char = []
    for d in data:
        _, q, x = d[:3]
        q_char = map(get_char_nums_for_word, q)
        x_char = map(get_char_nums_for_word, x)
        data_char.append([q_char, x_char])
    return data_char


train_char = prepare_chars(data_train)
dev_char = prepare_chars(data_dev)

print("Saving files to", out_path)

with open(join(out_path, 'train.json'), 'w') as f:
    json.dump(data_train, f)
with open(join(out_path, 'dev.json'), 'w') as f:
    json.dump(data_dev, f)

with open(join(out_path, 'train_words.json'), 'w') as f:
    json.dump(train_num, f)
with open(join(out_path, 'dev_words.json'), 'w') as f:
    json.dump(dev_num, f)

with open(join(out_path, 'train_bin_feats.json'), 'w') as f:
    json.dump(train_bin_feats, f)
with open(join(out_path, 'dev_bin_feats.json'), 'w') as f:
    json.dump(dev_bin_feats, f)

with open(join(out_path, 'train_char_ascii.json'), 'w') as f:
    json.dump(train_char, f)
with open(join(out_path, 'dev_char_ascii.json'), 'w') as f:
    json.dump(dev_char, f)
