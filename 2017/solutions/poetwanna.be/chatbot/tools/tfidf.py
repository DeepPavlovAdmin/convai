import cPickle
import numpy as np

import config
from tools import progbar


_idf = None
_avg_idf = None


def init_data():
    global _idf
    global _avg_idf

    if _idf is not None or not config.knn_apply_idf:
        return

    t = progbar.Timer('Initializing knn idfs...')
    with open(config.knn_idf, 'rb') as f:
        _idf = cPickle.load(f)
    for w, v in _idf.items():
        _idf[w] = np.power(v, 0.65)  # sqrt(v)
    _avg_idf = np.mean(_idf.values())
    t.tok('knn idf range is (min / avg / max): %.2f / %.2f / %.2f' % (
        np.min(_idf.values()), _avg_idf, np.max(_idf.values())))


def has_idf(w):
    init_data()
    return w in _idf


def idf(w):
    '''Computed on the dialogue pairs corpora'''
    init_data()
    return _idf.get(w, None)


def avg_idf():
    init_data()
    return _avg_idf
