import re
import string
import numpy as np

import config
from tools import tfidf
from tools import tokenizer
from tools.embeddings import word2vec


nonalpha_pattern = re.compile('([\W_]+|[0-9]+)')


def utt_vec(utt, ignore_nonalpha=True, ignore_nonascii=False,
            correct_spelling=True, normalize=True, print_debug=False):
    word2vec.init_data()
    ret = np.zeros(word2vec.word_vecs().shape[1])
    if ignore_nonalpha:
        utt = nonalpha_pattern.sub(' ', utt)
    if ignore_nonascii:
        printable = set(string.printable)
        utt = filter(lambda x: x in printable, utt).decode('utf-8')
    tokens = tokenizer.tokenize(utt, correct_spelling=False)
    if print_debug:
        print '  Tokens:', ' # '.join(tokens).encode('utf-8')
    for complex_word in word2vec.token_iterator(tokens):
        vec = word2vec.vec(complex_word)
        if config.knn_apply_idf:
            idf = tfidf.idf(complex_word) if tfidf.has_idf(
                complex_word) else tfidf.avg_idf()
            if np.isfinite(idf):
                vec = vec * idf
        ret += vec
    len_ = np.linalg.norm(ret)
    if normalize and not np.isclose(len_, 0.0):
        ret /= len_
    elif print_debug:
        print '  WARNING: Sentence embedded as an empty vector'
    return ret
