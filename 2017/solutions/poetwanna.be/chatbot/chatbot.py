from __future__ import print_function

import codecs
import datetime as dt
import json
import os
import pprint
import random
import re
import signal
import sys
import traceback

from argparse import ArgumentParser

import config
import data_manager
import interface
from utils import U, Singleton
from telegram_iface import TelegramInterface
from tools import colors
from tools import profanity
from tools import tokenizer
from tools.embeddings import word2vec
from tools import coref_resolver
from tools.preprocess import Preprocessor

import talker
import talker.used_talkers
import async.talker
import utils


def get_talker_names(args_talkers=None):
    env_talkers = os.environ.get('TALKERS')
    if args_talkers is not None and env_talkers is not None:
        raise ValueError('Specify talkers either by cmd argument'
                         'or env variable')
    if args_talkers is not None:
        return args_talkers.split(',')
    if env_talkers is not None:
        return env_talkers.split(',')
    used_talkers = utils.list_talkers(talker.used_talkers)
    talkers = [ut.__name__ for ut in used_talkers]
    return talkers


class HumanBot(object):

    version = '0.1 human_bot'

    def flush_old_sessions(self, minutes_old=60):
        pass

    def update_contexts(self, user_utt, bot_utt, session_key):
        pass

    def respond_to(self, utt, session_key):
        print('User:', utt.encode('utf-8'))
        return raw_input('Response:'), []


def filter_nsfw_utterances(all_utt_dicts, user_utt_dict):
    def tokenize(s):
        return tokenizer.tokenize(s.lower(), lowercase_first_in_sentence=False,
                                  correct_spelling=False)

    user_toks = set(tokenize(user_utt_dict['raw_utt']))
    badwords = profanity.badwords.difference(user_toks)

    # Filter out fapwords
    for d in all_utt_dicts:
        if any([t in badwords for t in tokenize(d['utt'])]):
            print("NSFW Filtering: %s" % (d, ))
            d['utt'] = u'<filtered>'
            d['score'] = -10


def postprocess_utt(utt):
    '''Apply post-processing before sending the reply'''
    try:
        utt = U(utt)
        # Uppercase the first letter
        utt = utt[0].upper() + utt[1:]

        # Add a period if ends with two letters or (space, letter)
        if utt[-1].isalpha() and (utt[-2].isalpha() or utt[-2] == u' '):
            utt += u'.'
    except Exception, e:
        print('Utterance post-processing failed with: %s' % unicode(e))
    return utt


def select_follow_up(follow_ups):
    possible_shows = []
    for talker_name, utt, score in follow_ups:
        if score is None or score < 0:
            continue
        if 0 < score < 1:
            if random.random() < score:
                possible_shows.append((talker_name, utt, 1.0))
        if 1 <= score:
            w_score = 1.0 + (score - 1.0) * config.talker_weight[talker_name]
            possible_shows.append((talker_name, utt, w_score))
    if not possible_shows:
        return None
    random.shuffle(possible_shows)
    possible_shows.sort(key=lambda fu: fu[-1])
    print(possible_shows)
    return possible_shows[-1]


class LoggingBot(Singleton):

    def __init__(self, keep_logs=True, **kwargs):
        super(LoggingBot, self).__init__(**kwargs)
        self.keep_logs = keep_logs
        self._log_fpath_prefix = os.path.join(
            config.cli_scoring_logs,
            str(dt.datetime.now()).replace(' ', '_').replace(':', '-'))

    def log_fpath(self, session_key):
        return '%s_%s.txt' % (self._log_fpath_prefix, session_key)

    def log_last_utts(self, session_key):
        d = {'dialogue_pair': self.get_session(session_key)['last_utts']}
        self.write_to_logfile(d, session_key)

    def log_best_utt_num(self, n, session_key):
        self.write_to_logfile({'best_utt_num': n}, session_key)
        '''The num should be 1-based'''

    def write_to_logfile(self, d, session_key):
        with codecs.open(self.log_fpath(session_key), 'a', 'utf-8') as f:
            f.write(json.dumps(d) + '\n')


class SimpleBot(LoggingBot):

    version = ('0.4 devel')

    def __init__(self, talker_names, use_celery=False,
                 corenlp_url=None,
                 **kwargs):
        super(SimpleBot, self).__init__(**kwargs)
        if corenlp_url is None:
            corenlp_url = 'http://' + os.environ.get('CORENLP') + ':9090'
        self.preprocessor = Preprocessor(corenlp_url)
        self.talker_names = talker_names
        self.sessions = {}
        if use_celery:
            self.talker_manager = CeleryTalkerManager(
                self.talker_names)
        else:
            self.talker_manager = TalkerManager(self.talker_names)

    def get_session(self, session_key):
        if session_key not in self.sessions:
            self.sessions[session_key] = {
                'start_date': dt.datetime.now(),
                'states': self.talker_manager.new_state(),
                'preproc_state': self.preprocessor.new_state(),
                'article': None,
                'last_utts': {
                    'user': None,
                    'bot': None,
                    'follow_up': None,
                    'all_talkers': None,
                }
            }
        return self.sessions[session_key]

    def flush_old_sessions(self, minutes_old=120):
        remove = []
        for key, sess in self.sessions.items():
            time_delta = dt.datetime.now() - sess['start_date']
            if time_delta.total_seconds() > minutes_old * 60:
                remove.append(key)
        for key in remove:
            del self.sessions[key]

    def set_article(self, text, session_key):
        # TODO(Adrian) Extract additional article metadata
        sess = self.get_session(session_key)

        (sess['preproc_state'], article_dict
         ) = self.preprocessor.set_article(sess['preproc_state'], text)

        sess['article'] = article_dict
        sess['states'] = self.talker_manager.set_article(
            sess['states'], article_dict)

        if self.keep_logs:
            d = {'article': article_dict, 'session_key': session_key}
            self.write_to_logfile(d, session_key)

    def respond_to(self, raw_utt, session_key):
        sess = self.get_session(session_key)
        self.flush_old_sessions()

        if type(raw_utt) is str:
            raw_utt = raw_utt.decode('utf-8')

        last_bot_utt = sess['last_utts']['bot']

        (sess['preproc_state'], user_utt_dict
         ) = self.preprocessor.preprocess(sess['preproc_state'],
                                          raw_utt, last_bot_utt)

        last_user_utt = sess['last_utts']['user']
        sess['last_utts']['user'] = user_utt_dict

        sess['states'], all_utt_dicts = self.talker_manager.get_responses(
            sess['states'], last_user_utt, last_bot_utt, user_utt_dict)

        filter_nsfw_utterances(all_utt_dicts, user_utt_dict)

        all_utt_dicts = sorted(all_utt_dicts, key=lambda rd: rd['score'] * -1)
        new_bot_utt = all_utt_dicts[0]['utt']

        # Add a follow-up maybe?
        sess['states'], follow_ups = self.talker_manager.get_follow_ups(
            sess['states'], all_utt_dicts)

        if config.debug:
            pprint.pprint({'all_utt_dicts': all_utt_dicts,
                           'follow_ups': follow_ups},
                          stream=sys.stderr)

        follow_ups = [(talker_name, utt, score)
                      for talker_name, (utt, score) in
                      zip(self.talker_names, follow_ups)]

        sess['last_utts']['follow_up'] = None
        fu = None
        if follow_ups:
            fu = select_follow_up(follow_ups)
        if fu:
            sess['last_utts']['follow_up'] = fu
            new_bot_utt = fu[1]

        new_bot_utt = postprocess_utt(new_bot_utt)

        for u in all_utt_dicts:
            u['utt'] = postprocess_utt(u['utt'])
        sess['last_utts']['bot'] = new_bot_utt
        sess['last_utts']['all_talkers'] = all_utt_dicts
        if self.keep_logs:
            self.log_last_utts(session_key)
        return new_bot_utt, all_utt_dicts


class TalkerManager(object):

    talker_weight = config.talker_weight

    def __init__(self, talker_names):
        self.talkers = []
        for tn in talker_names:
            talker_class = getattr(talker, tn)
            self.talkers.append((talker_class(async=False),
                                 self.talker_weight[tn]))

    def new_state(self):
        return [t.new_state() for t, _tw in self.talkers]

    def set_article(self, states, article_dict):
        new_states = []
        for (t, _tw), s in zip(self.talkers, states):
            new_s = t.set_article(s, article_dict)
            new_states.append(new_s)
        assert len(states) == len(new_states)
        return new_states

    def build_utt_dict(self, name, talker_weight, talker_utt, confidence):
        if type(talker_utt) is str:
            talker_utt = talker_utt.decode('utf-8')
        utt_dict = {'talker_name': name, 'utt': talker_utt,
                    'score': confidence * talker_weight,
                    'confidence': confidence, 'talker_weight': talker_weight}
        return utt_dict

    def get_responses(
            self, states, last_user_utt_dict, last_bot_utt, user_utt_dict):
        all_utt_dicts = []
        new_states = []
        for (t, tw), s in zip(self.talkers, states):
            new_states.append(s)
            try:
                (new_s, talker_utt, conf
                 ) = t.respond_to(
                    s, last_user_utt_dict, last_bot_utt, user_utt_dict)
                new_states[-1] = new_s
                all_utt_dicts.append(
                    self.build_utt_dict(t.name, tw, talker_utt, conf))
            except Exception, e:
                print('EXCEPTION:', unicode(e))
                _ex_type, _ex, tb = sys.exc_info()
                traceback.print_tb(tb)
        assert len(new_states) == len(self.talkers)
        return new_states, all_utt_dicts

    def get_follow_ups(self, states, new_bot_utt):
        follow_ups = []
        new_states = []
        for (t, _tw), s in zip(self.talkers, states):
            new_states.append(s)
            try:
                (new_s, fu, score) = t.follow_up(s, new_bot_utt)
                if type(fu) is str:
                    fu = fu.decode('utf-8')
                states[-1] = new_s
                follow_ups.append((fu, score))
            except Exception, e:
                print('EXCEPTION:', unicode(e))
                _ex_type, _ex, tb = sys.exc_info()
                traceback.print_tb(tb)
        follow_ups = [fu_score for fu_score in follow_ups
                      if None not in fu_score]
        assert len(new_states) == len(self.talkers)
        return new_states, follow_ups


class CeleryTalkerManager(TalkerManager):

    def __init__(self, talker_names):
        super(CeleryTalkerManager, self).__init__(talker_names=[])

        for tn in talker_names:
            talker_class = getattr(async.talker, tn)
            self.talkers.append((talker_class(async=True),
                                 self.talker_weight[tn]))

    def new_state(self):
        rets = [t.new_state.delay() for t, _tw in self.talkers]
        return [r.get() for r in rets]

    def set_article(self, states, article_dict):
        rets = [t.set_article.delay(s, article_dict)
                for (t, _tw), s in zip(self.talkers, states)]
        new_states = [r.get() for r in rets]
        assert len(states) == len(new_states)
        return new_states

    def get_responses(
            self, states, last_user_utt_dict, last_bot_utt, user_utt_dict):
        all_utt_dicts = []
        rets = [t.respond_to.delay(
            s, last_user_utt_dict, last_bot_utt, user_utt_dict)
            for (t, _tw), s in zip(self.talkers, states)]
        new_states = []
        for (t, tw), s, r in zip(self.talkers, states, rets):
            new_states.append(s)
            try:
                (new_s, talker_utt, conf
                 ) = r.get()
                new_states[-1] = new_s
                all_utt_dicts.append(
                    self.build_utt_dict(t.name, tw, talker_utt, conf))
            except Exception, e:
                print('EXCEPTION:', unicode(e))
                _ex_type, _ex, tb = sys.exc_info()
                traceback.print_tb(tb)
        assert len(new_states) == len(self.talkers)
        return new_states, all_utt_dicts

    def get_follow_ups(self, states, new_bot_utt):
        follow_ups = []
        rets = [t.follow_up.delay(s, new_bot_utt)
                for (t, _tw), s in zip(self.talkers, states)]
        new_states = []
        for (t, _tw), s, r in zip(self.talkers, states, rets):
            new_states.append(s)
            try:
                (new_s, fu, score) = r.get()
                if type(fu) is str:
                    fu = fu.decode('utf-8')
                states[-1] = new_s
                follow_ups.append((fu, score))
            except Exception, e:
                print('EXCEPTION:', unicode(e))
                _ex_type, _ex, tb = sys.exc_info()
                traceback.print_tb(tb)
        follow_ups = [fu_score for fu_score in follow_ups
                      if None not in fu_score]
        assert len(new_states) == len(self.talkers)
        return new_states, follow_ups


def opts_parser():
    usage = "Runs the chatbot with the desired interface"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-i', '--interface', default='convai',
        help=('Bot interface [convai|convai_test|convai_human|telegram|cli|'
              'cli_scoring] (default: %(default)s)'))
    parser.add_argument(
        '-t', '--talkers', default=None,
        help=('Specify talkers (as a comma-separated list, e.g., '
              'CraftedKnnTalker,SQUADTalker)'))
    parser.add_argument(
        '-m', '--use-multiprocessing',
        action='store_true', default=False,
        help=('Run talkers in separate processes, '
              '(default: %(default)s)'))
    return parser


if __name__ == "__main__":
    parser = opts_parser()
    args = parser.parse_args()

    def signal_handler(signal, frame):
        print(colors.reset_code())
        print('Caught SIGINT.')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Do it here rather than during first conversation
    word2vec.init_data(vocab=True, vectors=False)

    talkers = get_talker_names(args.talkers)

    if args.interface.lower() == 'cli':
        bot = SimpleBot(talker_names=talkers,
                        use_celery=args.use_multiprocessing)
        iface = interface.CliInterface(bot)
        iface.run()

    if args.interface.lower() == 'cli_scoring':
        bot = SimpleBot(talker_names=talkers,
                        use_celery=args.use_multiprocessing)
        iface = interface.CliInterface(bot, score_answers=True)
        iface.run()

    elif args.interface.lower() == 'telegram':
        raise NotImplementedError
        bot = SimpleBot(talker_names=talkers,
                        use_celery=args.use_multiprocessing)
        iface = TelegramInterface(bot)
        iface.run()

    elif args.interface.lower() == 'convai':
        bot = SimpleBot(talker_names=talkers,
                        use_celery=args.use_multiprocessing)
        iface = interface.ConvAIInterface(bot)
        iface.run()

    elif args.interface.lower() == 'convai_test':
        bot = SimpleBot(talker_names=talkers,
                        use_celery=args.use_multiprocessing)
        iface = interface.ConvAIInterface(
            bot, convai_url=config.convai_test_bot_url)
        iface.run()

    elif args.interface.lower() == 'convai_human':
        bot = HumanBot()
        iface = interface.ConvAIInterface(bot)
        iface.run()

    else:
        raise ValueError
