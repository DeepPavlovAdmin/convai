#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle


def to_unicode(x):
    return unicode(x, 'utf8')


def from_unicode(x):
    return x.encode('utf8')

text_converter = (lambda x: unicode(x, 'utf8'),
                  lambda x: x.encode('utf8'))
pickle_converter = (pickle.loads,
                    lambda x: pickle.dumps(x, pickle.HIGHEST_PROTOCOL))


class RedisDict(object):
    __slots__ = ['_db', '_prefix']
    to_value = {
        'text': text_converter,
        'corefed_text': text_converter,
        'raw_utt': text_converter,
        'spelled_utt': text_converter,
        'corefed_utt': text_converter,
        'spelled_tags': pickle_converter,
        'corefed_tags': pickle_converter,
    }

    def __init__(self, db, prefix):
        self._db = db
        self._prefix = prefix + '.'

    def __getitem__(self, key):
        return RedisDict.to_value[key][0](self._db.get(self._prefix + key))

    def __setitem__(self, key, val):
        return self._db.set(self._prefix + key,
                            RedisDict.to_value[key][1](val))

    def __getattribute__(self, name):
        ga = super(RedisDict, self).__getattribute__
        if name in RedisDict.__slots__:
            return ga(name)
        return RedisDict(
            ga('_db'), ga('_prefix') + name)
