from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import anydbm

from .posting_list_hex import PostingList

from utils import to_utf8


class CachedDB:

    def __init__(self, name, converter):
        print("CachedDB: opening", name)
        self.db = anydbm.open(name, 'r')
        self.converter = converter
        self.cache = {}

    def __contains__(self, key):
        key = to_utf8(key)
        return key in self.cache or key in self.db

    def __getitem__(self, key):
        key = to_utf8(key)
        if key in self.cache:
            return self.cache[key]

        value = self.converter(self.db[key])
        self.cache[key] = value

        return value


class IndexReader:

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
        self.cache[term] = result
        return result
