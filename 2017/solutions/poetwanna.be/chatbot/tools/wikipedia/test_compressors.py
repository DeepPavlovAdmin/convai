#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import snappy
import blosc
import datetime
import timeit

import numpy as np

from tools.wikipedia.posting_list_hex import PostingList


def blosc_compress(lists, **kwargs):
    ret = []
    for l in lists:
        l = np.concatenate(([0], l))
        l = np.diff(l)
        ret.append(blosc.compress(l, **kwargs))
    return ret


def blosc_decompress():
    ret = []
    d = blosc.decompress
    for l in blosc_compressed:
        l = d(l)
        f = np.fromstring
        c = np.cumsum
        ret.append(c(f(l, 'int64')))
    return ret


def hex_decompress():
    ret = []
    for l in hex_compressed:
        ret.append(PostingList(l).to_list())
    return ret


def compressed_size(lists):
    return sum([len(l) for l in lists])

if __name__ == '__main__':
    hex_compressed = []
    lists = []

    with open('/data/wikipedia/wiki3M/wiki.index.txt') as f:
        for _, l in zip(range(5000), f):
            l = l.strip().split(None, 1)[1]
            hex_compressed.append(l)
            lists.append(np.array(PostingList(l).to_list()))

    lref = np.concatenate(lists)

    print("Hex size:   %d, time: %s" %
          (compressed_size(hex_compressed),
           np.mean(timeit.repeat("hex_decompress()",
                                 "from __main__ import hex_decompress",
                                 number=3, repeat=3))))
    assert (np.all(lref == np.concatenate(hex_decompress())))

    kwargs = dict(cname='lz4hc')
    for cname in ['blosclz', 'snappy', 'lz4', 'lz4hc', 'zlib', 'zstd']:
        kwargs['cname'] = cname
        for clevel in range(10):
            kwargs['clevel'] = clevel
            blosc_compressed = blosc_compress(lists, **kwargs)
            print("Blosc size: %d, time: %s (%s" %
                  (compressed_size(blosc_compressed),
                   np.mean(timeit.repeat(
                       "blosc_decompress()",
                       "from __main__ import blosc_decompress",
                       number=3, repeat=3)),
                   kwargs,
                   ))
            assert (np.all(lref == np.concatenate(blosc_decompress())))
