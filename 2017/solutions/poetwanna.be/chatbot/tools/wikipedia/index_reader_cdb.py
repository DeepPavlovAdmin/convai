from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from blosc import decompress
import cdb
import mcdb
import numpy as np
import struct

from .posting_list_hex import PostingList

from utils import to_utf8
import os


def blosc_decompress_int_list(s):
    s = decompress(s)
    return np.cumsum(np.fromstring(s, np.int64))


class Missing():
    pass
_missing = Missing()


class CachedDB(object):

    def __init__(self, name, converter):
        super(CachedDB, self).__init__()
        mcdb_name = name[:-4] + '.mcdb'
        if os.path.exists(name):
            print("CDB: opening", name)
            self.db = cdb.init(name)
            self.db_contains = lambda key: self.db.has_key(key)
        elif os.path.exists(mcdb_name):
            print("MCDB: opening", mcdb_name)
            self.db = mcdb.read(mcdb_name)
            self.db_contains = lambda key: self.db.get(
                key, _missing) is not _missing
        else:
            raise ValueError("Unknown file: %s" % (name, ))
        if converter in [int, float]:
            if converter == int:
                s = struct.Struct('<Q').unpack
            else:
                s = struct.Struct('<f').unpack

            def c(v, s=s):
                return s(v)[0]
            self.converter = c
        elif converter == 'blosc_to_list':
            self.converter = blosc_decompress_int_list
        self.cache = {}

    def __contains__(self, key):
        key = to_utf8(key)
        return key in self.cache or self.db_contains(key)

    def __getitem__(self, key):
        key = to_utf8(key)
        if key in self.cache:
            return self.cache[key]

        value = self.converter(self.db[key])
        if len(self.cache) > 1e5:
            self.cache = {}
        self.cache[key] = value

        return value


class IndexReader(CachedDB):

    def __init__(self, index_name, index_positions_name):
        del index_positions_name  # unused
        super(IndexReader, self).__init__(
            index_name[:-4] + '.cdb', 'blosc_to_list')

    def get(self, term):
        try:
            return self[term]
        except KeyError:
            return set()


class IndexReaderHex:

    def __init__(self, index_name, index_positions_name):
        self.index_file = open(index_name, 'r')
        self.cache = {}

        self.positions = CachedDB(index_positions_name, int)

    def get(self, term):
        if term not in self.positions:
            return set()
        if term in self.cache:
            return self.cache[term]

        pos = self.positions[term]
        self.index_file.seek(pos)
        line = self.index_file.readline().split()
        line = ' '.join(line[1:])
        result = PostingList(line).to_list()
        if len(self.cache) > 1e5:
            self.cache = {}
        self.cache[term] = result
        return result
