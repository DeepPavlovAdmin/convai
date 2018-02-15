# -*- encoding: utf8 -*-

from __future__ import print_function

import sys
import os
import config
import numpy as np


vectors = {}


for x in open(os.path.join(config.swt_data_path, 'simple_title_vec.txt')):
    L = x.split()
    if len(L) == 2:
        continue
    w = L[0]
    v = np.array(L[1:], dtype=float)
    length = v.dot(v)**0.5
    vectors[w] = v / length


def best(word, K):
    v = vectors[word]
    L = sorted([(v.dot(vectors[w]), w) for w in vectors], reverse=True)
    return L[:K]


def follow_ups(word):
    if word not in vectors:
        return []
    res = best(word, 20)
    print(res)
    return res


if __name__ == "__main__":

    for x in sys.stdin:
        w = x.strip()
        if w in vectors:
            print(w)
            for v, b in best(w, 20)[1:]:
                print('   ', v, b)
            print()
