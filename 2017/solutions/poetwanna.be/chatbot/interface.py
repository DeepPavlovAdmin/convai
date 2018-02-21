"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function

import codecs
import datetime
import json
import os
import random
import requests
import time

import tabulate

import config
from tools import colors


def format_utt_dicts(utt_dicts, utt_line_wrap=None):
    '''Formats a list of all bot responses into table'''
    headers = ['name', 'scor', 'cnfd', 'wght', '#', 'utterance']
    rows = [[d['talker_name'], '%.2f' % d['score'], '%.2f' % d['confidence'],
             '%.2f' % d['talker_weight'], i + 1, d['utt']]
            for i, d in enumerate(utt_dicts)]
    if utt_line_wrap is not None:
        for i in xrange(len(rows) - 1, -1, -1):
            if '_talker' in rows[i][0]:
                rows[i][0] = rows[i][0].replace('_talker', '')
            while len(rows[i][-1]) > utt_line_wrap:
                suffix_len = len(rows[i][-1]) % utt_line_wrap
                if suffix_len == 0:
                    suffix_len = utt_line_wrap
                rows.insert(i + 1, ['' for r in rows[i][:-1]] +
                                   [rows[i][-1][-suffix_len:]])
                rows[i][-1] = rows[i][-1][:-suffix_len]
    return tabulate.tabulate(rows, headers=headers)


class ConvAIInterface(object):

    def __init__(self, bot, convai_url=config.convai_bot_url):
        if not config.convai_bot_id:
            print('WARNING: config.convai_bot_id is not set.')
        self.bot = bot
        self.convai_url = convai_url
        self.active_chat_ids = set()
        print("ConvAI bot starting with URL: ", convai_url)

    def chat_is_ongoing(self, chat_id):
        return chat_id in self.active_chat_ids

    def observe(self, m, chat_id):
        observation = None
        if not self.chat_is_ongoing(chat_id):
            if m['message']['text'].startswith('/start '):
                self.active_chat_ids.add(chat_id)
                observation = m['message']['text']
                print("Start new chat #%s" % chat_id)
            else:
                print("Dialog not started yet. Ignore message.")
        else:
            if m['message']['text'] == '/end':
                print("End chat #%s" % chat_id)
                self.active_chat_ids.remove(chat_id)
            else:
                print("Accept message as part of chat #%s" % chat_id)
                observation = m['message']['text']
        return observation

    def act(self, observation, chat_id):
        if not self.chat_is_ongoing(chat_id):
            print("Dialog not started yet. Do not act.")
            return

        if observation is None:
            print("Empty message. Do not act.")
            return

        message = {
            'chat_id': chat_id
        }

        if observation.startswith('/start '):
            self.bot.set_article(observation[7:], chat_id)
            print("Set the article")
            return None

        text, all_utt_dicts = self.bot.respond_to(observation, chat_id)
        print(format_utt_dicts(
            all_utt_dicts, utt_line_wrap=60).encode('utf-8'))

        data = {}
        if text == '':
            print("Decided to do not respond and wait for new message")
            return
        elif text == '/end':
            print("Decided to finish chat %s" % chat_id)
            self.active_chat_ids.remove(chat_id)
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

        message['text'] = json.dumps(data)
        return message

    def run(self):

        print('Starting communication loop.')
        while True:
            try:
                time.sleep(1)
                res = requests.get(os.path.join(self.convai_url, 'getUpdates'))

                if res.status_code != 200:
                    print(res.text)
                    res.raise_for_status()

                for i, m in enumerate(res.json()):
                    print("Process message (%d of %d) %s" %
                          (i + 1, len(res.json()), m))
                    chat_id = m['message']['chat']['id']
                    observation = self.observe(m, chat_id)
                    new_message = self.act(observation, chat_id)
                    if new_message is not None:
                        res = requests.post(
                            os.path.join(self.convai_url, 'sendMessage'),
                            json=new_message,
                            headers={'Content-Type': 'application/json'})
                        print("Send response to server (status code %d)." %
                              res.status_code)
                        if res.status_code != 200:
                            print(res.text)
                            res.raise_for_status()
            except Exception as e:
                print("Exception: {}".format(e))


class CliInterface(object):

    def __init__(self, bot, score_answers=False):
        self.bot = bot
        self.score_answers = score_answers
        if score_answers:
            self.bot.keep_logs = True
        self.articles = []
        try:
            with codecs.open('/data/wikipedia/sample_articles.txt',
                             'r', 'utf8') as f:
                self.articles = f.readlines()
        except:
            pass

    def run(self):
        if self.articles:
            article = random.sample(self.articles, 1)[0]
            print(colors.code(fg='green') + 'Article: ' + article + '\n')
        else:
            article = raw_input(
                colors.code(fg='green') + 'Article: ').decode('utf-8')
        self.bot.set_article(article, session_key='12345')

        while True:
            user_utt = raw_input(
                colors.code(fg='yellow') + 'Me: ').decode('utf-8')
            print(colors.reset_code())
            bot_utt, all_utt_dicts = self.bot.respond_to(
                user_utt, session_key='12345')
            print(format_utt_dicts(all_utt_dicts,
                                   utt_line_wrap=60).encode('utf-8'))
            print(colors.colorize(
                'Bot: ' + bot_utt, fg='green').encode('utf-8'))
            if self.score_answers:
                n = self._get_utt_number(
                    'Enter the number of the best answer: ',
                    len(all_utt_dicts))
                if n is not None:
                    self.bot.log_best_utt_num(n, session_key=12345)
                else:
                    print('Invalid number, not scoring.')

    def _get_utt_number(self, prompt, n):
        r = raw_input(prompt)
        try:
            if 1 <= int(r) <= n:
                return int(r)
        except:
            return None
