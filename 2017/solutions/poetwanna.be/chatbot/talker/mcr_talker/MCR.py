from __future__ import print_function

import codecs
import os
import config
import numpy as np

from scipy import io
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from nltk import word_tokenize, sent_tokenize, bleu
from nltk.translate.bleu_score import SmoothingFunction

from tools.profanity import badwords


tox_thresh = 0.8
repeats = False
remove_uppercased_quotes = True

wikiquotes_path = config.mcr_path
floatx = config.mcr_floatx
debug = config.debug

tox_path = wikiquotes_path + 'mcrtox.npy'
quotes_path = wikiquotes_path + 'quotes.txt'
idf_path = wikiquotes_path + 'idf.npy'
words_path = wikiquotes_path + 'wordlist.txt'


''' READ DATA '''


if not repeats:
    tf_idf_path = wikiquotes_path + 'tf_idf_no_repeats.mtx'
    dense_vectors_path = wikiquotes_path + 'dense_no_repeats.npy'
else:
    tf_idf_path = wikiquotes_path + 'tf_idf.mtx'
    dense_vectors_path = wikiquotes_path + 'dense.npy'

tox = np.load(tox_path)

glove_words = np.load(config.glove_dict_path)
glove_w_to_i = {w: i for i, w in enumerate(glove_words)}

# WARNING: glove_vectors are reused in TriviaTalker and SQUADTalker!
glove_vectors = np.load(config.glove_embs_path)

tf_idf = io.mmread(tf_idf_path).tocsr().astype(floatx)
idf = np.load(idf_path).astype(floatx)
dense_vectors = np.load(dense_vectors_path).astype(floatx)


def load_lines(p):
    with codecs.open(p, encoding='utf-8') as f:
        return [l[:-1] for l in f]


words = load_lines(words_path)
w_to_i = {w: i for i, w in enumerate(words)}
word_idf = dict(zip(words, idf))

quotes = load_lines(quotes_path)
tokenized_quotes = [word_tokenize(q.lower()) for q in quotes]

trigger = """
    abortion
    american
    americans
    catholic
    christ
    christian
    christianity
    church
    communism
    communist
    communists
    dead
    death
    devil
    die
    euthanasia
    exterminate
    exterminated
    extermination
    genocide
    god
    hang
    hanged
    holocaust
    islam
    islamist
    islamists
    jehovah
    jesus
    kill
    killed
    lgbt
    murder
    murdered
    muslim
    muslims
    pope
    satan
    terror
    terrorist
    terrorism
"""

badwords = badwords | set([x.strip() for x in trigger.split('\n')])
bad_inds = [i for (i, quote) in enumerate(tokenized_quotes)
            if any(w in badwords for w in quote)]

print("MCR vulg:", len(bad_inds))


def upp_quote(q):
    q = q.split()
    if not q:
        return True
    for i in range(len(q) - 1):
        if q[i + 1] != 'I' and q[i + 1][0].isupper() and q[i][-1] not in '.?!':
            return True
    return False


if remove_uppercased_quotes:
    upper_inds = [i for (i, q) in enumerate(quotes) if upp_quote(q)]
    print("MCR upper:", len(upper_inds))
    bad_inds = list(set(bad_inds + upper_inds))

tox_inds = list(np.where(tox > 0.8)[0])
print("MCR tox:", len(tox_inds))
bad_inds = list(set(bad_inds + tox_inds))

print("MCR total discarded:", len(bad_inds))


''' SPARSE '''


def sparse_vec(seq, repeat=repeats):
    vec = lil_matrix((1, len(words)), dtype=floatx)
    if not seq:
        return vec.tocsr()
    unknown = 0
    for w in set(seq) if not repeat else seq:
        if w in w_to_i:
            vec[0, w_to_i[w]] += 1
        else:
            unknown += 1

    return (vec.multiply(idf) * (len(seq) - unknown) / len(seq)).tocsr(
    ).astype(floatx)


def get_sparse_sims(vec):
    return cosine_similarity(tf_idf, vec)[:, 0]


''' DENSE '''


def dense_vec(seq, repeat=repeats):
    vec = np.zeros(300, dtype=floatx)
    if not seq:
        return vec
    unknown = 0
    for w in set(seq) if not repeat else seq:
        if w in glove_w_to_i and w in word_idf:
            vec += glove_vectors[glove_w_to_i[w]] * word_idf[w]
        else:
            unknown += 1
    return (vec * (len(seq) - unknown) / len(seq)).astype(floatx)


def get_dense_sims(vec):
    return 1 - cdist(dense_vectors, vec[np.newaxis], metric='cosine')[:, 0]


''' ARTICLE '''


def get_important_words(article, k):
    words = []
    idfs = []
    article = map(word_tokenize, sent_tokenize(article))

    max_art_idf = 0

    for sent in article:
        for i, w in enumerate(sent):
            wl = w.lower()
            if wl in w_to_i:
                val = idf[w_to_i[wl]]
                max_art_idf = max(max_art_idf, val)
                if w[0].isupper() and i:
                    val *= 1.5
                words.append(wl)
                idfs.append(val)
    words = np.array(words)
    idfs = np.array(idfs)
    idx = (-idfs).argsort()

    rich_words = {}
    for w, val in zip(words[idx], idfs[idx]):
        if w not in rich_words:
            rich_words[w] = 0.0
        rich_words[w] += val
        if len(rich_words) == k:
            break
    return rich_words, max_art_idf


def summarize_article(article, num_words=10):
    rich_words, max_art_idf = get_important_words(article, num_words)
    if debug:
        print(rich_words, max_art_idf)
    dense_v = np.zeros(300, dtype=floatx)
    sparse_v = lil_matrix((1, len(words)), dtype=floatx)
    if not rich_words:
        return dense_v, sparse_v.tocsr(), max_art_idf

    unknown_glove = 0
    for w, val in rich_words.items():
        sparse_v[0, w_to_i[w]] = val
        if w in glove_w_to_i:
            dense_v += glove_vectors[glove_w_to_i[w]] * val
        else:
            unknown_glove += 1

    dense_v = (dense_v * (num_words - unknown_glove) /
               num_words).astype(floatx)
    return dense_v, sparse_v.tocsr().astype(floatx), max_art_idf


''' AUXILARY FUNCTIONS '''


sf = SmoothingFunction()


def tokenize(s):
    if not isinstance(s, unicode):
        s = unicode(s, 'utf8')
    return word_tokenize(s.lower())


def max_idf(seq):
    return max([idf[w_to_i[w]] for w in seq if w in w_to_i] or [0])


def softmax(x):
    x = np.exp(x - np.max(x, keepdims=True))
    return x / x.sum()
