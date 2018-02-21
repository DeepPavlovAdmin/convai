"""
Copyright 2017 Reasoning & Learning Lab, McGill University

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
# encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
import requests
import os
import json
import time
import random
import collections
import model_selection
import config
conf = config.get_config()
import random
import emoji
import storage
import logging
import traceback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)

MAX_CONTEXT = 3

chat_history = {}
mSelect = model_selection.ModelSelection()


class ChatState:
    START = 0     # when we received `/start`
    END = 1       # when we received `/end`
    CHATTING = 2  # all other times


class ConvAIRLLBot:

    def __init__(self):
        self.chat_id = None
        self.observation = None
        self.ai = {}

    def observe(self, m):
        chat_id = m['message']['chat']['id']
        state = ChatState.CHATTING  # default state
        # New chat:
        if chat_id not in self.ai:
            if m['message']['text'].startswith('/start '):
                self.ai[chat_id] = {}
                self.ai[chat_id]['chat_id'] = chat_id
                self.ai[chat_id]['observation'] = m['message']['text']
                self.ai[chat_id]['context'] = collections.deque(
                    maxlen=MAX_CONTEXT)
                logging.info("Start new chat #%s" % self.chat_id)
                state = ChatState.START  # we started a new dialogue
                chat_history[chat_id] = []
            else:
                logging.info("chat not started yet. Ignore message")
        # Not a new chat:
        else:
            # Finished chat
            if m['message']['text'] == '/end':
                logging.info("End chat #%s" % chat_id)
                mSelect.clean(chat_id)  # remove mSelect data for this chat id
                del self.ai[chat_id]
                state = ChatState.END  # we finished a dialogue
            # Continue chat
            else:
                self.ai[chat_id]['observation'] = m['message']['text']
                logging.info("Accept message as part of chat #%s" % chat_id)
                if not m['message']['text'].startswith('/start'):
                    chat_history[chat_id].append(
                        {'text': m['message']['text'], 'sender': "human"})
        return chat_id, state

    def act(self, chat_id, state,  m):
        data = {}
        message = {
            'chat_id': chat_id
        }

        if chat_id not in self.ai:
            # Finish chat:
            if m['message']['chat']['id'] == chat_id and m['message']['text'] == '/end':
                logging.info("Decided to finish chat %s" % chat_id)
                data['text'] = '/end'
                data['evaluation'] = {  # let's have the best default value haha
                    'quality': 5,
                    'breadth': 5,
                    'engagement': 5
                }
                storage.store_data(chat_id, chat_history[chat_id])
                del chat_history[chat_id]
                message['text'] = json.dumps(data)
                return message
            else:
                logging.info("Dialog not started yet. Do not act.")
                return

        if self.ai[chat_id]['observation'] is None:
            logging.info("No new messages for chat #%s. Do not act." %
                         self.chat_id)
            return

        model_name = 'none'
        policyID = -1
        if state == ChatState.START:
            text = "Hello! I hope you're doing well. I am doing fantastic today! Let me go through the article real quick and we will start talking about it."
        else:
            # select from our models
            (text, context, model_name), policyID = mSelect.get_response(
                chat_id, self.ai[chat_id]['observation'], self.ai[chat_id]['context'])
            self.ai[chat_id]['context'] = context
        #texts = ['I love you!', 'Wow!', 'Really?', 'Nice!', 'Hi', 'Hello', '', '/end']
        #text = texts[random.randint(0, 7)]

        if text.strip() == '':
            logging.info("Decided to respond with random emoji")
            data = {
                'text': random.choice(emoji.UNICODE_EMOJI.keys()),
                'evaluation': 0,  # 0=nothing, 1=thumbs down, 2=thumbs up
                'policyID': policyID,
                'model_name': 'rand_emoji'
            }
        else:
            logging.info("Decided to respond with text: %s, model name %s, policyID %d" % (
                text, model_name, policyID))
            data = {
                'text': text,
                'evaluation': 0,  # 0=nothing, 1=thumbs down, 2=thumbs up
                'model_name': model_name,
                'policyID': policyID
            }

        message['text'] = json.dumps(data)
        data['sender'] = 'bot'
        chat_history[chat_id].append(data)
        return message


def main():

    BOT_ID = conf.bot_token  # !!!!!!! Put your bot id here !!!!!!!

    if BOT_ID is None:
        raise Exception('You should enter your bot token/id!')

    BOT_URL = os.path.join(conf.bot_endpoint, BOT_ID)

    bot = ConvAIRLLBot()
    print "loading models"
    mSelect.initialize_models()  # should we start this in a new thread instead..?

    while True:
        try:
            time.sleep(1)
            logging.debug("Get updates from server")
            res = requests.get(os.path.join(BOT_URL, 'getUpdates'))

            if res.status_code != 200:
                logging.info(res.text)
                res.raise_for_status()

            logging.debug("Got %s new messages" % len(res.json()))
            for m in res.json():
                state = ChatState.START  # assume new chat all the time
                # will become false when we call bot.observe(m), except when it's really a new chat
                while state == ChatState.START:
                    logging.info("Process message %s" % m)
                    # return chat_id & the dialogue state
                    chat_id, state = bot.observe(m)
                    # pass chat_id, dialogue state & message to act upon
                    new_message = bot.act(chat_id, state, m)
                    if new_message is not None:
                        print new_message
                        logging.info("Send response to server.")
                        res = requests.post(os.path.join(BOT_URL, 'sendMessage'),
                                            json=new_message,
                                            headers={'Content-Type': 'application/json'})
                        if res.status_code != 200:
                            logging.info(res.text)
                            res.raise_for_status()

            logging.debug("Sleep for 1 sec. before new try")
        except Exception as e:
            logging.error(e)
            traceback.format_exc()


if __name__ == '__main__':
    main()
