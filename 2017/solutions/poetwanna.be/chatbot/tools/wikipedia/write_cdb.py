#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tools.wikipedia.posting_list_hex import PostingList

import argparse
import blosc
import cdb
import mcdb
import numpy as np
import os
import progressbar
import struct


def compress_hex_list(l):
    l = np.concatenate(([0], l))
    l = np.diff(l)
    return blosc.compress(l, cname='lz4hc')


def hexstr_to_list(l):
    return PostingList(l).to_list()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-type', default='int')
    parser.add_argument('in_file')
    parser.add_argument('out_file')

    args = parser.parse_args()

    str_to_val, val_to_str = {
        'int': (int, struct.Struct('<Q').pack),
        'float': (float, struct.Struct('<f').pack),
        'prych_hex': (hexstr_to_list, compress_hex_list)
    }[args.val_type]

    if args.out_file.endswith('.mcdb'):
        db = mcdb.make(args.out_file)
    else:
        db = cdb.cdbmake(args.out_file, args.out_file + '.tmp')
    with open(args.in_file, 'r') as f:
        f.seek(0, os.SEEK_END)
        pb = progressbar.ProgressBar(maxval=f.tell())
        pb.start()
        f.seek(0)
        for l in f:
            k, v = l.strip().split(None, 1)
            v = val_to_str(str_to_val(v))
            db.add(k, v)
            pb.update(f.tell())
        pb.finish()
    db.finish()
