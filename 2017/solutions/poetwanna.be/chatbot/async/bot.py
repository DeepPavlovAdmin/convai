#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import celery
import os
import random

import chatbot
import config
import talker as sync_talker
import tools.preprocess as sync_preprocess
from utils import U

from async.celery import app
from async.redis import (
    db, last_bot_utt_key, new_user_utt_key, last_user_utt_key,
    handle_async_error, new_bot_utts_key, new_bot_followups_key,
    article_key)
import time


class CeleryRedisBaseWrapper(object):
    def __init__(self, klass, kwargs):
        super(CeleryRedisBaseWrapper, self).__init__()
        if hasattr(klass, 'name'):
            self.name = 'rasync_' + klass.name

        self.base_name = 'async.bot.%s.%s.' % (
            klass.__module__, klass.__name__)
        self.klass = klass
        self.kwargs = kwargs
        self._object = None
        self.state_suffix = '.state.' + self.base_name

    @property
    def object(self):
        if self._object is None:
            self._object = self.klass(**self.kwargs)
        return self._object

    def get_state(self, session_key, pipe=None):
        _pipe = pipe or db.pipeline()
        state_key = session_key + self.state_suffix
        _pipe.get(state_key)
        pret = _pipe.execute()
        state_pickle = pret.pop(-1)
        if state_pickle is None:
            print('Computning new state for: %s' % (self.base_name,))
            state = self.object.new_state()
        else:
            state = pickle.loads(state_pickle)
        if pipe is not None:
            return state, pret
        else:
            return state

    def save_state(self, session_key, state, pipe=None):
        state_key = session_key + self.state_suffix
        data = pickle.dumps(state, protocol=-1)
        if pipe is None:
            db.set(state_key, data)
        else:
            pipe.set(state_key, data)
            pipe.execute()


class CeleryRedisTalker(CeleryRedisBaseWrapper):
    def __init__(self, **kwargs):
        super(CeleryRedisTalker, self).__init__(**kwargs)

        @app.task(name=self.base_name + 'set_article')
        def set_article(session_key):
            try:
                if getattr(self.object.set_article, '_no_op', False):
                    return
                pipe = db.pipeline(transaction=False)
                pipe.get(article_key(session_key))
                state, pret = self.get_state(session_key, pipe)
                article = pickle.loads(pret[0])
                state = self.object.set_article(state, article)
                self.save_state(session_key, state)
            except:
                handle_async_error(session_key=session_key)
        self.set_article = set_article

        @app.task(name=self.base_name + 'respond_to')
        def respond_to(session_key):
            try:
                if getattr(self.object._respond_to, '_no_op', False):
                    return
                pipe = db.pipeline(transaction=False)
                pipe.get(last_user_utt_key(session_key))
                pipe.get(last_bot_utt_key(session_key))
                pipe.get(new_user_utt_key(session_key))
                state, pipe_ret = self.get_state(session_key, pipe)
                (last_user_utt_dict, last_bot_utt, user_utt_dict
                 ) = (pickle.loads(r) for r in pipe_ret)
                state, bot_utt, confidence = self.object.respond_to(
                    state, last_user_utt_dict, last_bot_utt, user_utt_dict)

                if confidence is None or bot_utt is None:
                    self.save_state(session_key, state)
                    return

                name = self.klass.__name__
                weight = AsyncBot.talker_weight[name]
                score = confidence * weight
                utts = [{'talker_name': name,
                         'utt': chatbot.postprocess_utt(bot_utt),
                         'score': score,
                         'confidence': confidence,
                         'talker_weight': weight}]
                if self.klass.apply_profanity:
                    chatbot.filter_nsfw_utterances(utts, user_utt_dict)

                pipe = db.pipeline(transaction=False)
                pipe.zadd(new_bot_utts_key(session_key),
                          pickle.dumps(utts[0], -1), -utts[0]['score'])
                self.save_state(session_key, state, pipe)
            except:
                handle_async_error(session_key=session_key)
        self.respond_to = respond_to

        @app.task(name=self.base_name + 'follow_up')
        def follow_up(session_key):
            try:
                if getattr(self.object.follow_up, '_no_op', False):
                    return
                pipe = db.pipeline(transaction=False)
                pipe.zrange(new_bot_utts_key(session_key), 0, -1)
                state, pret = self.get_state(session_key, pipe)
                new_bot_utts = [pickle.loads(bu) for bu in pret[0]]

                state, new_bot_utt, confidence = self.object.follow_up(
                    state, new_bot_utts)

                pipe = db.pipeline(transaction=False)
                if confidence is not None and confidence > 0:
                    fu_data = pickle.dumps(
                        (self.klass.__name__, new_bot_utt, confidence), -1)
                    pipe.zadd(new_bot_followups_key(session_key),
                              fu_data, -confidence)
                self.save_state(session_key, state, pipe)
            except:
                handle_async_error(session_key=session_key)
        self.follow_up = follow_up


async_talkers = {}


for __n in dir(sync_talker):
    __t = getattr(sync_talker, __n)
    __is_talker = False
    try:
        if issubclass(__t, sync_talker.base.ResponderRole):
            __is_talker = True
    except:
        pass
    if not __is_talker:
        continue
    async_talkers[__n] = CeleryRedisTalker(
        klass=__t, kwargs=dict(async=True))


class CeleryRedisPreprocessor(CeleryRedisBaseWrapper):
    def __init__(self, **kwargs):
        super(CeleryRedisPreprocessor, self).__init__(
            klass=sync_preprocess.Preprocessor,
            kwargs=dict(
                corenlp_url='http://' + os.environ.get('CORENLP') + ':9090'),
            **kwargs)

        @app.task(name=self.base_name + 'set_article')
        def set_article(article, session_key):
            try:
                state = self.get_state(session_key)
                state, article_dict = self.object.set_article(state, article)
                pipe = db.pipeline(transaction=False)
                pipe.set(article_key(session_key),
                         pickle.dumps(article_dict, -1))
                self.save_state(session_key, state, pipe)
            except:
                handle_async_error(session_key=session_key)
        self.set_article = set_article

        @app.task(name=self.base_name + 'preprocess')
        def preprocess(user_raw_utt, session_key):
            try:
                pipe = db.pipeline(transaction=False)
                pipe.get(last_bot_utt_key(session_key))
                state, pret = self.get_state(session_key, pipe)
                last_bot_utt = pickle.loads(pret[0])

                state, user_utt_dict = self.object.preprocess(
                    state, user_raw_utt, last_bot_utt)

                pipe = db.pipeline(transaction=False)
                pipe.set(new_user_utt_key(session_key),
                         pickle.dumps(user_utt_dict, -1))
                self.save_state(session_key, state, pipe)
            except:
                handle_async_error(session_key=session_key)
        self.preprocess = preprocess

_preproc = CeleryRedisPreprocessor()


def Preprocessor():
    return _preproc


class AsyncBot(object):
    talker_weight = config.talker_weight
    none_pickle = pickle.dumps(None, -1)
    preprocessor = Preprocessor()

    @staticmethod
    def init_all():
        pass

    @staticmethod
    def async_set_article(session_key, article, talker_names):
        article = U(article)
        # set a failsafe version of the article
        db.set(article_key(session_key), pickle.dumps(
            {'text': article, 'corefed_text': article}, -1))
        talkers = [
            async_talkers[tn] for tn in talker_names if not
            getattr(async_talkers[tn].klass.set_article, '_no_op', False)]
        sa = dict(immutable=True)
        if config.celery_timeouts:
            sa['soft_time_limit'] = config.talker_article_timeout
            sa['time_limit'] = config.talker_article_timeout + 2
        job = (
            AsyncBot.preprocessor.set_article.signature(
                (article, session_key), **sa) |
            celery.group(t.set_article.signature((session_key,), **sa)
                         for t in talkers)
        )
        return job.delay()

    @staticmethod
    def async_respond_to_queued_utts(session_key, user_utt, talker_names):
        user_utt = U(user_utt)

        failsafe_user_utt = {
            'raw_utt': user_utt,
            'spelled_utt': user_utt,
            'spelled_tags': [],
            'corefed_utt':  user_utt,
            'corefed_tags': []
        }

        failsafe_bot_response = {
            'talker_name': "failsafe",
            'utt': u'Sorry, could you say that again, please :)',
            'score':-1.0,
            'confidence':-1.0,
            'talker_weight': 1.0
        }

        pipe = db.pipeline()
        pipe.set(last_bot_utt_key(session_key),
                 AsyncBot.none_pickle, nx=True)
        pipe.set(new_user_utt_key(session_key),
                 AsyncBot.none_pickle, nx=True)
        pipe.rename(new_user_utt_key(session_key),
                    last_user_utt_key(session_key))
        pipe.set(new_user_utt_key(session_key),
                 pickle.dumps(failsafe_user_utt, -1))
        pipe.delete(new_bot_utts_key(session_key))
        pipe.zadd(new_bot_utts_key(session_key),
                  pickle.dumps(failsafe_bot_response, -1),
                  -failsafe_bot_response['score'])
        pipe.delete(new_bot_followups_key(session_key))
        pipe.execute()

        responders = [
            async_talkers[tn] for tn in talker_names if not
            getattr(async_talkers[tn].klass._respond_to, '_no_op', False)]
        follow_uppers = [
            async_talkers[tn] for tn in talker_names if not
            getattr(async_talkers[tn].klass.follow_up, '_no_op', False)]

        sa = dict(immutable=True)
        if config.celery_timeouts:
            sa['soft_time_limit'] = config.talker_respond_timeout
            sa['time_limit'] = config.talker_respond_timeout + 2
        job = (
            AsyncBot.preprocessor.preprocess.signature(
                (user_utt, session_key), **sa) |
            celery.group(t.respond_to.signature((session_key,), **sa)
                         for t in responders) |
            # add noop because of https://github.com/celery/celery/issues/3585
            AsyncBot.noop.si(session_key)
        )
        if follow_uppers:
            job = (
                job | celery.group(t.follow_up.signature((session_key,), **sa)
                                   for t in follow_uppers))
        job = job | AsyncBot.combine_responses_and_follow_ups.signature(
            (session_key,), **sa)
        return job.delay(time_limit=3)

    @staticmethod
    @app.task(ignore_results=True)
    def noop(session_key):
        pass

    @staticmethod
    @app.task
    def combine_responses_and_follow_ups(session_key):
        pipe = db.pipeline()
        pipe.zrange(new_bot_utts_key(session_key), 0, -1)
        pipe.delete(new_bot_utts_key(session_key))
        pipe.zrange(new_bot_followups_key(session_key), 0, -1)
        pipe.delete(new_bot_followups_key(session_key))
        utts, _, follow_ups, _ = pipe.execute()

        utts = [pickle.loads(u) for u in utts]
        follow_ups = [pickle.loads(fu) for fu in follow_ups]

        new_bot_utt = utts[0]['utt']

        utt_table = []
        sel_fu = None
        if follow_ups:
            sel_fu = chatbot.select_follow_up(follow_ups)
        if sel_fu:
            name, fu, confidence = sel_fu
            new_bot_utt = chatbot.postprocess_utt(fu)
            utt_table.append({'talker_name': name + ' sel_fup',
                              'utt': fu,
                              'score': confidence,
                              'confidence': confidence,
                              'talker_weight': 1.0})

        for name, fu, confidence in follow_ups:
            score = confidence
            if confidence and 1 < confidence:
                score = 1.0 + (score - 1.0) * config.talker_weight[name]
            utt_table.append({'talker_name': name + ' fup',
                              'utt': fu,
                              'score': score,
                              'confidence': confidence,
                              'talker_weight': 1.0})

        utt_table.extend(utts)
        db.set(last_bot_utt_key(session_key), pickle.dumps(new_bot_utt, -1))
        return new_bot_utt, utt_table

    @staticmethod
    def get_utterances_so_far(session_key):
        utts = [pickle.loads(u) for u in
                db.zrange(new_bot_utts_key(session_key), 0, -1)]
        if utts:
            return utts[0]['utt'], utts

    @staticmethod
    def get_debug_information(session_key):
        pickled_user_utt = db.get(new_user_utt_key(session_key))
        if pickled_user_utt:
            preprocessed = pickle.loads(pickled_user_utt)
            ret = [(k, preprocessed[k]) for k in
                   ['raw_utt', 'spelled_utt', 'corefed_utt']]
            return ret
        return ''

    @staticmethod
    @app.task
    def init_squad_task():
        sync_talker.SQUADTalker(async=True)

    @staticmethod
    def init_squad(n_tasks=10):
        return [AsyncBot.init_squad_task.delay()
                for _ in range(n_tasks)]

    @staticmethod
    def wait_squad(tasks):
        while tasks:
            time.sleep(1)
            print ("Waiting for Squad init...")
            _t = []
            for t in tasks:
                if t.ready():
                    t.get()
                else:
                    _t.append(t)
            tasks = _t
