#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import celery

import config
import time

from .redis import log_time
import collections
import re
import traceback
import sys
from tools.embeddings import word2vec
import utils

redis_address = 'redis://' + config.redis + ':6379/0'

app = celery.Celery('async',
                    broker=redis_address,
                    backend=redis_address,
                    include=[
                       'async.wikipedia',
                       'async.talker',
                       'async.bot',
                       ]
                    )

app.conf.task_routes = ([
# # for easy debug you can put a talker into a new queue
#     ('async.*tools.*', {'queue': 'testing',
#                         'delivery_mode': 'transient'}),
    ('async.wikipedia.*', {'queue': 'wiki',
                           'delivery_mode': 'transient'}),
    ('async.*.talker.alice_talker.*', {'queue': 'alice',
                                       'delivery_mode': 'transient'}),
#     ('async.*.talker.squad_talker.*', {'queue': 'squad',
#                                        'delivery_mode': 'transient'}),
#     ('async.*.talker.squad_brainy_trivia_talker.*', {'queue': 'squad',
#                                                      'delivery_mode': 'transient'}),
#     ('async.bot.init_squad_task', {'queue': 'squad',
#                                    'delivery_mode': 'transient'}),
#     ('async.*.talker.simple_wiki_talker.*', {'queue': 'simple_wiki',
#                                              'delivery_mode': 'transient'}),
#     ('async.*.talker.brainy_smurf_talker.*', {'queue': 'simple_wiki',
#                                               'delivery_mode': 'transient'}),
#     ('async.*.talker.*', {'queue': 'talkers',
#                           'delivery_mode': 'transient'}),
    ('*', {'queue': 'default',
           'delivery_mode': 'transient'}),
],)


worker_queues = {
#     'worker_squad': ['squad'],
#     'worker_simple_wiki': ['simple_wiki'],
    'worker_simple_talkers': ['talkers', 'default'],
    'worker_alice': ['alice'],
    'worker_wiki': ['wiki'],
}


app.config_from_object('async.celeryconfig')


def revoke_all(r):
    q = [r]
    while q:
        r = q.pop()
        if r.children:
            q.extend(r.children)
        if r.parent:
            q.append(r.parent)
        r.revoke()


@celery.signals.celeryd_after_setup.connect
def setup_direct_queue(sender, instance, **kwargs):
    worker_type = sender.split('@')[0]
    for queue_name in worker_queues[worker_type]:
        print('%s adding queue %s' % (worker_type, queue_name))
        instance.app.amqp.queues.select_add(queue_name)


@celery.signals.worker_init.connect()
def configure_workers(sender=None, **kwargs):
    try:
        # to set worker nameuse the -n argument!
        worker_type = sender.hostname.split('@')[0]
        if worker_type == 'worker_wiki':
            from tools.wikipedia import Wikipedia
            Wikipedia(config.default_wiki)
            return
        word2vec.init_data(vocab=True, vectors=False)

        handled_talkers = set()
        import talker.used_talkers
        used_talkers = list(utils.list_talkers(talker.used_talkers))

        if worker_type == 'test_worker':
            for talker_class in utils.list_talkers(talker.used_talkers):
                try:
                    talker_class(async=True)
                except:
                    _ex_type, _ex, tb = sys.exc_info()
                    traceback.print_tb(tb)
                    print("Exception in ", talker_class)

        if not config.preload_talkers:
            return

        # Instantiate all talkers, with the exception of SQUAD with CUDA here:
        talker_re = re.compile(r'.*.talker\.([a-z_]+).*')
        for route_re, route_opts in app.conf.task_routes[0]:
            m = talker_re.match(route_re)
            if not m:
                continue
            talker_mod = m.group(1)
            for talker_class in used_talkers:
                if talker_mod in talker_class.__module__:
                    handled_talkers.add(talker_class)
                    if route_opts['queue'] in worker_queues[worker_type]:
                        # this is "our talker"
                        load = True
                        if utils.which('nvidia-smi') and 'squad' in talker_mod:
                            # suqd doesn't like multiprocessing
                            load = False
                        if load:
                            try:
                                talker_class(async=True)
                            except:
                                _ex_type, _ex, tb = sys.exc_info()
                                traceback.print_tb(tb)
                                print("Exception in ", talker_class)

        if worker_type == 'worker_simple_talkers':
            for talker_class in used_talkers:
                if talker_class not in handled_talkers:
                    try:
                        talker_class(async=True)
                    except:
                        _ex_type, _ex, tb = sys.exc_info()
                        traceback.print_tb(tb)
                        print("Exception in ", talker_class)

    except:
        _ex_type, _ex, tb = sys.exc_info()
        traceback.print_tb(tb)
        print("Exception in worker_init")
    sys.stdout.flush()
    sys.stderr.flush()

if config.debug:
    timings_d = collections.defaultdict(lambda: 0)

    @celery.signals.task_prerun.connect
    def task_prerun_handler(**kwargs):
        timings_d[kwargs['task'].name] += time.time()

    @celery.signals.task_postrun.connect
    def task_postrun_handler(**kwargs):
        cost = time.time() - timings_d.pop(kwargs['task'].name)
        if kwargs['task'].name.startswith('async.bot.'):
            try:
                session_key = kwargs['kwargs'].get('session_key', kwargs['args'][-1])
            except:
                session_key = 'all_sessions'
        else:
            session_key = 'all_sessions'
        log_time(session_key, kwargs['task'].name, cost)

if __name__ == '__main__':
    app.start()
