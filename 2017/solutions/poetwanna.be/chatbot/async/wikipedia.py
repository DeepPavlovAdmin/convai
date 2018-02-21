#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from async.redis import cache_it
from async.celery import app
from tools.wikipedia import Wikipedia as SyncWikipedia
import config

task_args = {}
if config.celery_timeouts:
    task_args['soft_time_limit'] = config.wiki_timeout
    task_args['time_limit'] = config.wiki_timeout + 2


@app.task(**task_args)
def find_paragraphs(wiki_name, Q, K, main_title=None, main_title_bonus=1.,
                    debug=False):
    return SyncWikipedia(wiki_name).find_paragraphs(
        Q, K, main_title=main_title, main_title_bonus=main_title_bonus,
        debug=debug)


@app.task(**task_args)
def get_main_article(wiki_name, Q, K, debug=False):
    return SyncWikipedia(wiki_name).get_main_article(Q, K, debug=debug)


@app.task(**task_args)
def find_titles(wiki_name, Q, K, debug=False):
    return SyncWikipedia(wiki_name).find_titles(Q, K, debug=debug)


class Wikipedia(object):
    def __init__(self, name):
        self.name = name

    @cache_it()
    def find_paragraphs(self, *args, **kwargs):
        r = find_paragraphs.delay(self.name, *args, **kwargs)
        return r.get(disable_sync_subtasks=False)

    @cache_it()
    def get_main_article(self, *args, **kwargs):
        r = get_main_article.delay(self.name, *args, **kwargs)
        return r.get(disable_sync_subtasks=False)

    @cache_it()
    def find_titles(self, *args, **kwargs):
        r = find_titles.delay(self.name, *args, **kwargs)
        return r.get(disable_sync_subtasks=False)
