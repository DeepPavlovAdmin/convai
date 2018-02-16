#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import pprint
import random
import select
import sys
import uuid
import time


from async.bot import AsyncBot
from async.redis import get_timings, db, get_async_errors
from interface import format_utt_dicts
from tools import colors
from utils import Singleton


from chatbot import get_talker_names, LoggingBot
from argparse import ArgumentParser
import config
import requests
import os
import json
from async import celery, redis
import utils


user_nags = [
    u"Hi, let's talk! Ask me questions!",
    u"I know lots of things",
    u"Don't keep me waiting for you",
    u"I desperately need you to talk to me!",
    u"Shh, good you are silent, the Zombies are coming, help me!",
    u"Hello, help! It's them!",
    u"Help!",
    u"Braaaaains!!",
]


class Interface(LoggingBot):

    def __init__(self, talker_names=None, wait_for_workers=True,
                 print_timings=False, gather_stats=False,
                 nag_user=True, **kwargs):
        super(Interface, self).__init__(**kwargs)
        self.articles = []
        self.computations = {}
        self.nag_user = nag_user
        self.user_nag_timer = {}
        self.unprocessed_utts = collections.defaultdict(lambda: [])
        if talker_names is None:
            talker_names = get_talker_names()
        self.talker_names = talker_names
        self.print_timings = print_timings
        self.gather_stats = gather_stats
        self.stats = collections.defaultdict(lambda: 0)

        need_workers = set(celery.worker_queues.keys())
        if wait_for_workers:
            squads = AsyncBot.init_squad()
            while True:
                time.sleep(1)
                has_workers = celery.app.control.inspect().ping()
                if has_workers is None:
                    has_workers = set()
                else:
                    has_workers = set(w.split('@')[0] for w in has_workers)
                missing_workers = need_workers.difference(has_workers)
                if not missing_workers:
                    break
                print("Waiting for workers: %s\nRunning workers: %s" %
                      (missing_workers, has_workers))
            AsyncBot.wait_squad(squads)

    def log_dialogue_pair(self, session_key, user_utt, bot_utts):
        if not self.keep_logs or user_utt is None or not bot_utts:
            return
        try:
            d = {'dialogue_pair': {
                'user': user_utt,
                'bot': bot_utts[0],
                'all_talkers': bot_utts[1],
            }}
            self.write_to_logfile(d, session_key)
        except:
            print(colors.colorize(utils.get_ex_info(), fg="red"))

    def set_user_nag_timer(self, session_key, bump=0):
        if not self.nag_user:
            return
        last_msg = self.user_nag_timer.get(session_key, (None, 0))
        # Nag the guy in 30 s
        timeout = 60 * 1.5**(min(8, last_msg[1]))
        self.user_nag_timer[session_key] = (time.time() + timeout,
                                            last_msg[1] + bump)

    def check_user_nags(self):
        t = time.time()
        for session_key, (nag_time, nag_cnt) in self.user_nag_timer.items():
            if t > nag_time:
                nag = user_nags[min(len(user_nags) - 1, nag_cnt)]
                print(colors.colorize(
                    "Nagging the user in %s" % (session_key,), fg="blue"))
                yield (session_key, (nag, []))
                self.set_user_nag_timer(session_key, bump=1)

    def remove_session(self, session_key):
        self.user_nag_timer.pop(session_key, None)
        comp = self.computations.pop(session_key, None)
        if comp:
            celery.revoke_all(comp)

    def async_set_article(self, session_key, article):
        self.set_user_nag_timer(session_key)

        if session_key in self.computations:
            return False
        self.computations[session_key] = AsyncBot.async_set_article(
            session_key, article, self.talker_names)
        self.computations[session_key]._t_start = time.time()
        self.computations[session_key]._user_utt = None

        if self.keep_logs:
            try:
                d = {'article': {'text': article}, 'session_key': session_key}
                self.write_to_logfile(d, session_key)
            except:
                print(colors.colorize(utils.get_ex_info(), fg="red"))
        return True

    def async_handle_user_utt(self, session_key, user_utt):
        if user_utt:
            self.unprocessed_utts[session_key].append(user_utt)
        if session_key in self.computations:
            return False

        # protection from key error on .pop() for empty user_utt
        self.unprocessed_utts[session_key]

        full_user_utt = ' '.join(self.unprocessed_utts.pop(session_key))
        if not full_user_utt:
            return True

        self.computations[session_key
                          ] = AsyncBot.async_respond_to_queued_utts(
            session_key, full_user_utt, self.talker_names)
        self.computations[session_key]._t_start = time.time()
        self.computations[session_key]._user_utt = full_user_utt
        return True

    def get_responses(self):
        new_comps = {}
        ready = []
        to_start = []
        for session_key, comp in self.computations.items():
            if not comp.ready():
                if (time.time() - comp._t_start >
                        config.global_response_timeout):
                    celery.revoke_all(comp)
                    print(colors.colorize(
                        "Removing stale computation for %s" %
                        (session_key), fg='red'))
                    # try to salvage a result!
                    utt = AsyncBot.get_utterances_so_far(session_key)
                    self.log_dialogue_pair(
                        session_key, comp._user_utt, utt)
                    if utt:
                        ready.append((session_key, utt))
                        self.set_user_nag_timer(session_key)
                else:
                    new_comps[session_key] = comp
            else:
                try:
                    res = comp.get()
                except:
                    try:
                        res = AsyncBot.get_utterances_so_far(session_key)
                    except:
                        res = None
                    print(colors.colorize(utils.get_ex_info(), fg='red'))
                self.log_dialogue_pair(
                    session_key, comp._user_utt, res)

                if 0:
                    print(colors.colorize("Got async results: %s" % (res,),
                                          fg='blue'))
                else:
                    print(colors.colorize("Got async results for %s" %
                                          (session_key,), fg='blue'))

                if session_key in self.unprocessed_utts:
                    to_start.append(session_key)
                else:
                    # Start nagging timer
                    self.set_user_nag_timer(session_key)

                if res and res[0] is not None:
                    # Only return reponses, not results of set article
                    # and similar crap
                    ready.append((session_key, res))
                for timings_key in [session_key, 'all_sessions']:
                    timings = get_timings(timings_key)  # TODO XXX
                    if timings and self.print_timings:
                        print(colors.colorize(
                            'Total task time in session %s %f:' %
                            (timings_key, sum((t[1] for t in timings)),),
                            fg='blue'))
                        print(colors.colorize(
                            pprint.pformat(timings, depth=2, width=60),
                              fg='blue'))
                for errors_key in [session_key, 'all_sessions']:
                    errors = get_async_errors(errors_key)
                    if errors:
                        print(colors.colorize(
                            'Errors during query %s:\n%s' %
                            (timings_key, u'\n'.join(errors),),
                            fg='red'))

                print(colors.colorize('Bot took %f s' %
                                      (time.time() - comp._t_start),
                                      fg='blue'))
        self.computations = new_comps
        for session_key in to_start:
            ret = self.async_handle_user_utt(session_key, '')
            assert ret
        ready.extend(self.check_user_nags())
        return ready

    def print_response(self, all_utt_dicts, bot_utt):
        if self.gather_stats:
            for ud in all_utt_dicts:
                if ud['talker_name'].endswith('sel_fup'):
                    self.stats['_followup_count'] += 1
                    self.stats[ud['talker_name']] += 1
                elif not ud['talker_name'].endswith('fup'):
                    self.stats['_response_count'] += 1
                    self.stats[ud['talker_name']] += 1
                    break  # don't record other bots

        print(format_utt_dicts(
            all_utt_dicts, utt_line_wrap=60).encode('utf-8'))
        print(colors.colorize('Bot: ' + bot_utt, fg='green').encode('utf-8'))

    def print_stats(self):
        for k, v in sorted(self.stats.items()):
            if k.endswith('sel_fup'):
                print("%s: %f (%d/%d)" %
                      (k, 100. * v / self.stats['_followup_count'],
                       v, self.stats['_followup_count']))
            elif not k.startswith('_'):
                print("%s: %f (%d/%d)" %
                      (k, 100. * v / self.stats['_response_count'],
                       v, self.stats['_response_count']))


class ConvaiInterface(Interface):

    def __init__(self, convai_url, **kwargs):
        super(ConvaiInterface, self).__init__(**kwargs)
        self.convai_url = convai_url
        self.active_chat_ids = set()
        self.orig_chat_ids = {}
        print("ConvAI bot starting with URL: ", convai_url)

    def chat_is_ongoing(self, chat_id):
        return chat_id in self.active_chat_ids

    def remove_session(self, session_key):
        self.active_chat_ids.remove(session_key)
        self.orig_chat_ids.pop(session_key, None)
        super(ConvaiInterface, self).remove_session(session_key)

    def respond_with(self, chat_id, text):
        data = {}
        orig_chat_id = self.orig_chat_ids[chat_id]
        if text == '':
            print("Decided to do not respond and wait for new message")
            return
        elif text == '/end':
            print("Decided to finish chat %s" % chat_id)
            self.remove_session(chat_id)
            data['text'] = '/end'
            data['evaluation'] = {
                'quality': 0,
                'breadth': 0,
                'engagement': 0
            }
        else:
            print("Decided to respond with text: %s" % text)
            data = {
                'text': text,
                'evaluation': 0
            }
        message = {
            'chat_id': orig_chat_id,
            'text': json.dumps(data)
        }
        res = requests.post(os.path.join(self.convai_url, 'sendMessage'),
                            json=message,
                            headers={'Content-Type': 'application/json'})
        print("Send response to server (status code %d)." % res.status_code)
        if res.status_code != 200:
            print(res.text)
            res.raise_for_status()

    def process_chat_message(self, m):
        chat_id = m['message']['chat']['id']
        self.orig_chat_ids[str(chat_id)] = chat_id
        chat_id = str(chat_id)
        text = m['message']['text']
        if not self.chat_is_ongoing(chat_id):
            if text.startswith('/start '):
                self.active_chat_ids.add(chat_id)
                print("Start new chat #%s" % chat_id)
                if self.async_set_article(chat_id, text[7:]):
                    print("Start async article processing")
                else:
                    print("Async article processing failed article ignored")
            else:
                print("Dialog not started yet. Ignore message.")
        else:
            if text == '/end':
                print("End chat #%s" % chat_id)
                self.remove_session(chat_id)
            else:
                print("Accept message as part of chat #%s" % chat_id)
                if self.async_handle_user_utt(chat_id, text):
                    print("Started async reply")
                else:
                    print("Message queued for async reply")

    def loop(self):
        for chat_id, (text, _) in self.get_responses():
            self.respond_with(chat_id, text)

        res = requests.get(os.path.join(self.convai_url, 'getUpdates'))

        if res.status_code != 200:
            print(res.text)
            res.raise_for_status()

        for i, m in enumerate(res.json()):
            print("Process message (%d of %d) %s" %
                  (i + 1, len(res.json()), m))
            self.process_chat_message(m)

    def run(self):
        print('Starting communication loop.')
        while True:
            try:
                self.loop()
                time.sleep(1)
            except Exception as e:
                print("Exception: {}".format(e))


class CliInterface(Interface):

    def __init__(self, wikinews=True, **kwargs):
        super(CliInterface, self).__init__(**kwargs)
        try:
            with codecs.open('/data/wikipedia/sample_articles.txt',
                             'r', 'utf8') as f:
                self.articles = f.readlines()
        except:
            pass
        self.wikinews = wikinews

    def run(self):
        session_key = str(uuid.uuid4())
        if self.wikinews:
            article = utils.fetch_random_wikinews()
            print(colors.code(fg='green') + 'Article: ' + article + '\n')
        elif self.articles:
            article = random.sample(self.articles, 1)[0]
            print(colors.code(fg='green') + 'Article: ' + article + '\n')
        else:
            article = raw_input(
                colors.code(fg='green') + 'Article: ').decode('utf-8')
        print(colors.code(fg='purple') +
              "Running an async bot, you can type stuff in between :)" +
              colors.code(fg='yellow'))
        sys.stdout.flush()

        comp_started = self.async_set_article(session_key, article)
        assert comp_started

        timeout = 1
        while True:
            # sleep 1 s waiting for user input
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)

            for _sk, res in self.get_responses():
                assert _sk == session_key
                if config.debug:
                    pprint.pprint(
                        AsyncBot.get_debug_information(session_key))
                self.print_response(res[1], res[0])

            if rlist:
                utt = sys.stdin.readline().strip()
                comp_started = self.async_handle_user_utt(session_key, utt)

                # internal buchaltery to inform the user on what is going on
                if comp_started:
                    print(colors.code(fg='purple') +
                          "Running an async bot, you can type stuff in between"
                          " :)" + colors.code(fg='yellow'))
                else:
                    print(colors.code(fg='purple') +
                          "The bot is running, the computation is delayed\n" +
                          "So far got:\n" +
                          str(self.unprocessed_utts[session_key]) +
                          "\n" + colors.code(fg='yellow'))
                sys.stdout.flush()


class FileInterface(Interface):

    def __init__(self, utt_file, wikinews=True, **kwargs):
        kwargs.setdefault('gather_stats', True)
        super(FileInterface, self).__init__(**kwargs)
        try:
            with codecs.open('/data/wikipedia/sample_articles.txt',
                             'r', 'utf8') as f:
                self.articles = f.readlines()
        except:
            pass
        self.wikinews = wikinews
        self.utt_file = utt_file

    def run(self):
        session_prefix = str(uuid.uuid4())
        art_cnt = 1
        session_key = session_prefix + str(art_cnt)
        for line in codecs.open(self.utt_file, encoding='utf8'):
            line = line.strip()
            if not line:
                continue
            while session_key in self.computations:
                for _sk, res in self.get_responses():
                    assert _sk == session_key
                    if config.debug:
                        pprint.pprint(
                            AsyncBot.get_debug_information(session_key))
                    self.print_response(res[1], res[0])
                time.sleep(1)
            if line.lower().startswith('article: '):
                art_cnt += 1
                self.remove_session(session_key)
                session_key = session_prefix + str(art_cnt)
                article = line[len('article: '):]
                print(colors.code(fg='green') + 'Article: ' + article + '\n')
                if article:
                    comp_started = self.async_set_article(session_key, article)
                    assert comp_started
            else:
                utt = line
                print(colors.colorize('User: ' + utt, fg='yellow'))
                comp_started = self.async_handle_user_utt(session_key, utt)
                assert comp_started
        if self.gather_stats:
            self.print_stats()


if __name__ == "__main__":
    usage = "Runs the chatbot with the desired interface"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-i', '--interface', default='cli',
        help=('Bot interface [cli]'
              ' (default: %(default)s)'))
    parser.add_argument(
        '-w', action='store_true', default=False,
        help=('Wait for workers'))
    parser.add_argument(
        '-t', '--talkers', default=None,
        help=('Specify talkers (as a comma-separated list, e.g., '
              'CraftedKnnTalker,SQUADTalker)'))
    parser.add_argument(
        '--timings', action='store_true', default=False,
        help=('Print timings'))
    parser.add_argument(
        '-f', '--file', default=None,
        help=('Read user utteraces from file. Used with CLI interface'))

    args = parser.parse_args()
    talker_names = get_talker_names(args.talkers)
    print("Using talkers", talker_names)
    kwargs = dict(talker_names=talker_names,
                  wait_for_workers=args.w,
                  print_timings=args.timings)

    if args.interface.lower() == 'file' or args.file:
        db.flushdb()
        iface = FileInterface(utt_file=args.file,
                              **kwargs)
        iface.run()

    if args.interface.lower() == 'cli':
        db.flushdb()
        iface = CliInterface(**kwargs)
        iface.run()
    if args.interface.lower() == 'convai_test':
        db.flushdb()
        iface = ConvaiInterface(convai_url=config.convai_test_bot_url,
                                keep_logs=False, **kwargs)
        iface.run()
    if args.interface.lower() == 'convai':
        iface = ConvaiInterface(convai_url=config.convai_bot_url,
                                keep_logs=False, **kwargs)
        iface.run()
