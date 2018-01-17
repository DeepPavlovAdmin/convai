import zmq
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import requests
import os
import json
import time
import random
import collections
import config
conf = config.get_config()
import random
import emoji
import numpy as np
# import storage
from model_selection_q import ModelID, ModelSelectionAgent
import multiprocessing
from Queue import Queue
from threading import Thread
import logging
from datetime import datetime
import traceback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)

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


MAX_CONTEXT = 3

chat_history = {}
chat_timing = {} # just for testing response time

# Queues
processing_msg_queue = Queue()
outgoing_msg_queue = Queue() #multiprocessing.JoinableQueue()

# Pipes
# parent to bot caller
PARENT_PIPE = 'ipc:///tmp/parent_push.pipe'
# bot to parent caller
PARENT_PULL_PIPE = 'ipc:///tmp/parent_pull.pipe'


class ChatState:
    START = 0     # when we received `/start`
    END = 1       # when we received `/end`
    CHATTING = 2  # all other times
    CONTROL = 3   # for control msgs


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
                self.ai[chat_id]['context'] = [] # changed from deque since it is not JSON serializable
                self.ai[chat_id]['allowed_model'] = ModelID.ALL
                logging.info("Start new chat #%s" % self.chat_id)
                state = ChatState.START  # we started a new dialogue
                chat_history[chat_id] = []
                logging.info("started new chat with {}".format(chat_id))
            else:
                logging.info("chat not started yet. Ignore message")
        # Not a new chat:
        else:
            # Finished chat
            if m['message']['text'] == '/end':
                logging.info("End chat #%s" % chat_id)
                processing_msg_queue.put({'control': 'clean', 'chat_id': chat_id})
                del self.ai[chat_id]
                state = ChatState.END  # we finished a dialogue
            # TODO: Control statement for allowed models
            # Statement could start with \model start <model_name>
            # End with \model end <model_name>
            elif m['message']['text'].startswith("\\model"):
                controls = m['message']['text'].split()
                if controls[1] == 'start':
                    # always provide a valid model name for debugging.
                    self.ai[chat_id]['allowed_model'] = controls[2]
                    logging.info("Control msg accepted, selecting model {}"
                            .format(controls[2]))
                else:
                    self.ai[chat_id]['allowed_model'] = ModelID.ALL
                    logging.info("Control msg accepted, resetting model selection")
                state = ChatState.CONTROL
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
        if chat_id not in self.ai:
            # Finish chat:
            if m['message']['chat']['id'] == chat_id and m['message']['text'] == '/end':
                logging.info("Decided to finish chat %s" % chat_id)
                data['text'] = '/end'
                data['evaluation'] = {
                    'quality': 5,
                    'breadth': 5,
                    'engagement': 5
                }
                if chat_id in chat_history:
                    # storage.store_data(chat_id, chat_history[chat_id])
                    del chat_history[chat_id]
                outgoing_msg_queue.put(({'data': data, 'chat_id': chat_id},{}))
                return
            else:
                logging.info("Dialog not started yet. Do not act.")
                return

        if self.ai[chat_id]['observation'] is None:
            logging.info("No new messages for chat #%s. Do not act." %
                         self.chat_id)
            return

        model_name = 'none'
        policyID = -1
        if state != ChatState.CHATTING:
            if state == ChatState.CONTROL:
                text = "--- Control command received ---"
            if state == ChatState.START:
                text = "Hello! I hope you're doing well. I am doing fantastic today! Let me go through the article real quick and we will start talking about it."
            # push this response to `outgoing_msg_queue`
            outgoing_msg_queue.put(
                ({'text': text, 'chat_id': chat_id,
                    'model_name': model_name, 'policyID': policyID},{}))
        else:
            # send the message to process queue for processing
            processing_msg_queue.put({
                'chat_id': chat_id,
                'text': self.ai[chat_id]['observation'],
                'context': self.ai[chat_id]['context'],
                'allowed_model': self.ai[chat_id]['allowed_model']
            })


# Initialize
BOT_ID = conf.bot_token  # !!!!!!! Put your bot id here !!!!!!!

if BOT_ID is None:
    raise Exception('You should enter your bot token/id!')

BOT_URL = os.path.join(conf.bot_endpoint, BOT_ID)

bot = ConvAIRLLBot()
model_selection_agent = None

def producer():
    """ call model selection agent here
    """
    while True:
        msg = processing_msg_queue.get()
        producer_process(msg)
        #m_p = multiprocessing.Process(target=producer_process, args=(msg,outgoing_msg_queue,
        #    model_selection_agent,))
        #m_p.daemon = True
        #m_p.start()
        processing_msg_queue.task_done()

def producer_process(msg):
    if 'text' in msg and 'chat_id' in msg and 'context' in msg:
        time_now = datetime.now()
        response = model_selection_agent.get_response(msg['chat_id'], msg['text'], msg['context'])
        if 'control' in msg and msg['control'] == 'test':
            logging.info("Response finally generated:")
            logging.info(msg['text'])
            resp_time = (time_now - chat_timing[msg['chat_id']]).total_seconds()
            logging.info("Resp time : {}".format(resp_time))
        else:
            outgoing_msg_queue.put((response,{
                'article_nouns': model_selection_agent.article_nouns,
                'chat_history': model_selection_agent.chat_history,
                'boring_count': model_selection_agent.boring_count,
                'used_models': model_selection_agent.used_models,
                'candidate_model': model_selection_agent.candidate_model
            }))

def response_receiver(telegram=True):
    """Receive response from either Telegram or console.
       Make its own thread for clarity
    """
    if telegram:
        while True:
            time.sleep(1)
            logging.debug("Get updates from server")
            res = requests.get(os.path.join(BOT_URL, 'getUpdates'))

            if res.status_code != 200:
                logging.info(res.text)
                logging.error('Remote Network issue, system idle')
            else:
                logging.debug("Got %s new messages" % len(res.json()))
                for m in res.json():
                    state = ChatState.START  # assume new chat all the time
                    # will become false when we call bot.observe(m),
                    # except when it's really a new chat
                    while state == ChatState.START:
                        logging.info("Process message %s" % m)
                        # return chat_id & the dialogue state
                        chat_id, state = bot.observe(m)
                        bot.act(chat_id, state, m)
    else:
        # TODO: implement later
        pass


def reply_sender():
    """ Send reply to Telegram or console
        Thread: Read from `outgoing_msg_queue` and send to server
    """
    while True:
        package = outgoing_msg_queue.get()
        msg, cache = package[0], package[1]
        # preserving the state
        #for key,value in cache.iteritems():
        #    setattr(model_selection_agent, key, value)
        outgoing_msg_queue.task_done()
        if 'chat_id' not in msg:
            continue
        chat_id = msg['chat_id']
        message = {
            'chat_id': chat_id
        }
        data = {}
        if 'data' in msg:
            data = msg['data']
        else:
            text = msg['text']
            model_name = msg['model_name']
            policyID = msg['policyID']
            if 'context' in msg:
                bot.ai[chat_id]['context'] = msg['context'][-MAX_CONTEXT:]

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
        # if chat has ended, then no need to put the chat history
        if chat_id in chat_history:
            chat_history[chat_id].append(data)

        logging.info("Send response to server.")
        res = requests.post(os.path.join(BOT_URL, 'sendMessage'),
                            json=message,
                            headers={'Content-Type': 'application/json'})
        if res.status_code != 200:
            logging.info(res.text)
            res.raise_for_status()


def stop_app():
    """Broadcast stop signal
    """
    processing_msg_queue.put({'control': 'exit'})

def test_app():
    """ Perform heavy testing on the app.
    Send msgs and log the response avg response time.
    """
    chat_id = random.randint(1,100000)
    text = random.choice(['ok, how was it', 'wow good observation',
        'where did it happen?', 'can you explain?', 'ok', 'hmm', 'good',
        'i dont like this', 'interesting', 'where did you see this?',
        'who made it?','\start McGill University is a public research organization'])
    context = ['preset context']
    allowed_model = 'all'
    processing_msg_queue.put({
        'chat_id': chat_id,
        'text': text,
        'context': context,
        'allowed_model': allowed_model,
        'control' : 'test'
    })
    chat_timing[chat_id] = datetime.now()
    

if __name__ == '__main__':
    """ Start the threads.
    1. Response reciever thread
    2. Producer thread. bot -> model_selection
    3. Consumer thread. model_selection -> bot
    4. Reply thread. bot -> Telegram
    """
    MODE = 'production' # can be 'test' or anything else
    model_selection_agent = ModelSelectionAgent()

    response_receiver_thread = Thread(target=response_receiver, args=(True,))
    response_receiver_thread.daemon = True
    response_receiver_thread.start()
    producer_thread = Thread(target=producer)
    producer_thread.daemon = True
    producer_thread.start()
    reply_thread = Thread(target=reply_sender)
    reply_thread.daemon = True
    reply_thread.start()

    logging.info("====================================")
    logging.info("======RLLCHatBot Active=============")
    logging.info("All modules of the bot have been loaded.")
    logging.info("Thanks for your patience")
    logging.info("-------------------------------------")
    logging.info("Made with <3 in Montreal")
    logging.info("Reasoning & Learning Lab, McGill University")
    logging.info("Fall 2017")
    logging.info("=====================================")

    try:
        while True:
            if MODE == 'test':
                test_app()
                if random.choice([True, False]):
                    test_app()
                test_app()
                #test_app()
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Stopping model response selector")
        stop_app()
        logging.info("Closing app")
