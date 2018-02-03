from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import collections
import config
conf = config.get_config()
from models.wrapper import HRED_Wrapper
import logging
import model_selection
import json
import random

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
        )

MAX_CONTEXT = 3
mSelect = model_selection.ModelSelection()
conv_articles = json.load(open('data/articles.json','r'))

def help_cmd(bot, update):
    _check_user(update.message.chat_id)
    text = 'ConvAI demo session'
    bot.send_message(chat_id=update.message.chat_id, text=text)

def start_twitter(bot, update):
    _check_user(update.message.chat_id)
    _reset_user(update.message.chat_id)
    ai.history[update.message.chat_id]['model'] = 'twitter'
    text = 'I am now using my Twitter model to chat with you.'
    bot.send_message(chat_id=update.message.chat_id, text=text)

def start_reddit(bot, update):
    _check_user(update.message.chat_id)
    _reset_user(update.message.chat_id)
    ai.history[update.message.chat_id]['model'] = 'reddit'
    text = 'I am now using my Reddit model to chat with you.'
    bot.send_message(chat_id=update.message.chat_id, text=text)

def start_conv(bot, update):
    _check_user(update.message.chat_id)
    _reset_user(update.message.chat_id)
    ai.history[update.message.chat_id]['model'] = 'started'
    text = random.choice(conv_articles)
    bot.send_message(chat_id=update.message.chat_id, text=text)
    text, context = mSelect.get_response(update.message.chat_id,'/start ' + text,
            ai.history[update.message.chat_id]['context'])
    ai.history[update.message.chat_id]['context'] = context
    bot.send_message(chat_id=update.message.chat_id, text=text)

def reset_context(bot, update):
    _check_user(update.message.chat_id)
    _reset_user(update.message.chat_id)
    bot.send_message(chat_id=update.message.chat_id, text='Okay, let\'s start over.')

def get_response(bot, update):
    _check_user(update.message.chat_id)
    text, context = mSelect.get_response(update.message.chat_id, update.message.text,
                                        ai.history[update.message.chat_id]['context'])
    ai.history[update.message.chat_id]['context'] = context
    bot.send_message(chat_id=update.message.chat_id, text=text)

class Bot(object):

    def __init__(self):
        # For each user, we should keep track of their history as well as which bot they are talking to.
        self.history = {}
        self.updater = Updater(conf.test_bot_token)
        self.dp = self.updater.dispatcher

        self._add_cmd_handler('start', start_conv)
        self._add_cmd_handler('help', help_cmd)
        #self._add_cmd_handler('starttwitter', start_twitter)
        #self._add_cmd_handler('startreddit', start_reddit)
        #self._add_cmd_handler('resetcontext', reset_context)

        msg_handler = MessageHandler(Filters.text, get_response)
        self.dp.add_handler(msg_handler)

        self.models = {}

    def _add_cmd_handler(self, name, fn):
        handler = CommandHandler(name, fn)
        self.dp.add_handler(handler)

    def power_on(self):
        print 'Bot listening...'
        self.updater.start_polling()
        self.updater.idle()


def _reset_user(user_id):
    ai.history[user_id]['context'] = collections.deque(maxlen=MAX_CONTEXT)

def _check_user(user_id):
    # Check if we have the user in our memory, if not... start talking with Twitter bot.
    if user_id not in ai.history:
        ai.history[user_id] = {'context': collections.deque(maxlen=MAX_CONTEXT)}

if __name__ == '__main__':
    ai = Bot()
    mSelect.initialize_models()
    ai.power_on()
