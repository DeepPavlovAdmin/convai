#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import cPickle as pickle
import redis
import redis_cache

import config
import time
import utils

db = redis.Redis(host=config.redis, db=1)

# Caching
_cache = redis_cache.SimpleCache(host=config.redis, db=1)


def cache_it(limit=10000, expire=60 * 15):
    return redis_cache.cache_it(limit=limit, expire=expire,
                                cache=_cache)


def atomic_read_del_list(key, converter=None):
    pipe = db.pipeline()
    pipe.lrange(key, 0, -1)
    pipe.delete(key)
    vals, _ = pipe.execute()
    if converter is not None:
        vals = [converter(v) for v in vals]
    return vals


# Location of stuff in Redis
def article_key(session_key):
    return session_key + '.article'


def last_user_utt_key(session_key):
    return session_key + '.last_user_utt'


def last_bot_utt_key(session_key):
    return session_key + '.last_bot_utt'


def new_user_utt_key(session_key):
    return session_key + '.new_user_utt'


def new_bot_utts_key(session_key):
    return session_key + '.new_bot_utts'


def new_bot_followups_key(session_key):
    return session_key + '.new_bot_followups'


def timings_key(session_key):
    return session_key + '._timings'


def log_time(session_key, name, elapsed):
    db.rpush(timings_key(session_key),
             pickle.dumps((name, elapsed)))


@contextmanager
def redis_timer(session_key, name):
    t_start = time.time()
    yield
    elapsed = time.time() - t_start
    log_time(session_key, name, elapsed)


def get_timings(session_key):
    return atomic_read_del_list(timings_key(session_key), pickle.loads)


def errors_key(session_key):
    return session_key + '._errors'


def handle_async_error(session_key='all_sessions',
                       also_print=config.debug):
    def act(session_key=session_key, also_print=also_print):
        err_str = utils.get_ex_info()
        if also_print:
            print(err_str)
        db.lpush(errors_key(session_key), err_str)

    if config.debug:
        act()
    else:  # Really silence all errors in production run!
        try:
            act()
        except:
            pass


def get_async_errors(session_key='all_sessions'):
    return atomic_read_del_list(errors_key(session_key),
                                lambda s: unicode(s, 'utf8'))
