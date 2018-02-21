from __future__ import print_function

# score scaling based on idf, tags and capital letters

import numpy as np
import config

from pattern.en import verbs as vb


knn_idf_path = config.knn_idf
word_idf = None
max_idf = None

punc = set('~`!@#$%^&*()_+-={[}]<>,.?/;:"\'')

banned_verbs = 'do have be'.split()


def is_verb(w, tag):
    return tag.startswith('VB') and vb.conjugate(w) not in banned_verbs


def is_noun(w, tag):
    return tag.startswith('NN')


def is_adj(w, tag):
    return tag.startswith('JJ')

scr_power = 1.


def idf_score_modifier(tags, capital_bonus=1.25, nouns=True, verbs=False,
                       adjs=False):
    global word_idf
    if word_idf is None:
        word_idf = np.load(knn_idf_path)
        global max_idf
        max_idf = max(word_idf.values())

    if tags and tags[0][1] == config.unknown_tag:
        print('unknown_tag, returning 1.')
        return 1.

    question_tokens = [x[0] for x in tags]
    tags = [x[1] for x in tags]

    idfs = []

    for i, t in enumerate(question_tokens):
        tag = tags[i]
        tl = t.lower()
        t_idf = None
        skip_word = True

        if tl in punc:
            continue
        if nouns and is_noun(tl, tag):
            skip_word = False
        elif verbs and is_verb(tl, tag):
            skip_word = False
        elif adjs and is_adj(tl, tag):
            skip_word = False

        if not skip_word:
            if t in word_idf:
                t_idf = word_idf[t]
            elif tl in word_idf:
                t_idf = word_idf[tl]
            else:
                t_idf = max_idf

        if t_idf is not None:
            t_idf = t_idf**scr_power
            if t[0].isupper() and i:
                t_idf *= capital_bonus

            idfs.append(t_idf)

    if not idfs:
        return 0.

    denom = max_idf**scr_power * capital_bonus
    res = max(idfs) / denom

    print(question_tokens, tags, res)

    return res
