import codecs
import numpy as np
from collections import deque
from itertools import islice
from scipy.spatial.distance import cdist

import config
from tools import progbar


_word_vecs = None
_word_inds = None
_inds_word = None


def init_data(vocab=True, vectors=True):
    global _word_vecs
    global _word_inds
    global _inds_word
    if vocab and _word_inds is None:
        t = progbar.Timer('Loading word2vec (vocab)...')
        _word_inds = {}
        _inds_word = {}
        # Do not use codecs.open, it will falsely iterate over lines in the file
        # treating some chars as \n
        with open(config.word2vec_txt, 'r') as f:
            for i, word in enumerate(f):
                word = word.decode('utf-8').strip().strip('#')
                if (len(word) > 0 and
                        not _word_inds.has_key(word) and
                        (len(word) > 1 or word[0].isalnum())):
                    _word_inds[word] = i
                    _inds_word[i] = word
        t.tok('loaded %d words' % len(_word_inds))

    if vectors and _word_vecs is None:
        floatx = config.word2vec_floatx
        t = progbar.Timer('Loading word2vec (vectors, %s)...' % floatx)

        if config.word2vec_load_all_to_ram:
            _word_vecs = np.fromfile(config.word2vec_vecs, dtype=floatx)
            _word_vecs = word_vecs.reshape(-1, 300)
        else:
            _word_vecs = np.load(config.word2vec_vecs_small).astype(floatx)

        if config.word2vec_normalize:
            _word_vecs /= np.sqrt(np.sum(_word_vecs ** 2, axis=1)[:, None])
        t.tok()

        # Sanity check; fails due to normalization before/after changing
        # precision
        # assert np.allclose(
        #     _word_vecs[1], _fetch_vec_from_file(1).astype(floatx))


def token_iterator(tokens):
    '''Iterates over words in tokens list

    Tries to build complex words before looking for embeddings,
    e.g., ['Michael', 'Jordan'] will be embedded as one vector
    for Michael_Jordan.

    '''
    # Try to combine tokens into complex words,
    # i.e., Barack Obama into Barack_Obama
    # (who is embedded as a single vector).
    # There are as many as 4 words combined like that
    it = iter(xrange(len(tokens)))
    for pos in it:
        for i in xrange(4, 0, -1):
            if pos + i <= len(tokens):
                complex_word = '_'.join(tokens[pos:pos + i])
                if has_vec(complex_word):
                    yield complex_word
                    # Advance iterator to skip matched words
                    _consume(it, i - 1)
                    break


def vector_iterator(tokens):
    '''Iterates over word embeddings

    Tries to build complex words before looking for embeddings,
    e.g., ['Michael', 'Jordan'] will be embedded as one vector
    for Michael_Jordan.

    '''
    for complex_word in token_iterator(tokens):
        yield vec(complex_word)


def _consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def word_vecs():
    init_data(vocab=False, vectors=True)
    return _word_vecs


def word_inds():
    init_data(vocab=True, vectors=False)
    return _word_inds


def inds_word():
    init_data(vocab=True, vectors=False)
    return _inds_word


def vec(w):
    init_data(vocab=True, vectors=True)
    ind = _word_inds.get(w, None)
    if ind is None:
        return None

    if ind < _word_vecs.shape[0]:
        return _word_vecs[ind]
    else:
        return _fetch_vec_from_file(ind)


def has_vec(w):
    init_data(vocab=True, vectors=True)
    return _word_inds.has_key(w)


def has_word(w):
    init_data(vocab=True, vectors=False)
    return _word_inds.has_key(w)


def _fetch_vec_from_file(ind):
    vector_size = 300
    binary_len = np.dtype(np.float32).itemsize * vector_size
    fpath_bin = config.word2vec_vecs

    ret = []
    with open(fpath_bin, 'rb') as fin:
        fin.seek(ind * binary_len)
        v = np.fromstring(fin.read(binary_len), dtype=np.float32)

    if config.word2vec_normalize:
        v /= np.sqrt(np.sum(v ** 2))
    return v


def find_closest_words(word, k):
    """
    Returns list of words closest to given word. The number of
    returned words is <= k (its < k  when some of the closest
    matches were filtered out).
    """
    if word not in _word_inds:
        print "[word2vec] word not found:", word
        return []
    word_vec = vec(word).reshape(1, -1)
    inds, dists = _get_kNN_indices(k, _word_vecs, word_vec)
    return zip(_indices_to_words(inds), dists)


def _get_kNN_indices(k, matrix, vector, metric='cosine'):
    distances = cdist(matrix, vector, metric).reshape(-1)
    indices = distances.argpartition(k)[:k]
    return indices, np.take(distances, indices)


def _indices_to_words(indices):
    index_word = inds_word()
    return [index_word[i] for i in indices if i in index_word]
