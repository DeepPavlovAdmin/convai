# -*- coding: utf-8 -*-

import collections
import config
import logging
import os
import random
import string
from ast import literal_eval
from time import sleep
import configparser
from telegram import ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram.parsemode import ParseMode
import sys
# sys.path.append(os.getcwd())

# from bot_code.state_tracker import StateTracker, StoriesHandler

TOKEN = config.telegram_bot["TOKEN"]
CONTEXT_SIZE = config.telegram_bot["CONTEXT_SIZE"]
REPLY_HIST_SIZE = config.telegram_bot["REPLY_HIST_SIZE"]
LOGFILE = config.telegram_bot["LOGFILE"]

# Enable logging
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOGFILE)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.

class TelegramInterface:

    def __init__(self, bot):
        self.history = {}

        self.updater = Updater(TOKEN)
        self.name = str(self).split(' ')[-1][:-1]

        self.dp = self.updater.dispatcher

        self.dp.add_handler(CommandHandler("start", start))
        self.dp.add_handler(CommandHandler("help", help))

        self.dp.add_handler(MessageHandler([Filters.text], echo))

        self.dp.add_error_handler(error)
        self.stories = StoriesHandler()
        logger.info('I\'m alive!')

        self.bot = bot

    def run(self):
        # Start the Bot
        self.updater.start_polling()

        # Run the bot until the you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.updater.idle()


def mess2dict(mes):
    return literal_eval(str(mes))


def start(tel_obj, update):
    md = mess2dict(update.message)
    sender_id = str(md['from']['id'])
    story = iface.stories.get_one()
    tel_obj.sendMessage(update.message.chat_id, text=story)
    self.bot.set_story(story, sender_id)
    iface.history[sender_id] = {  # "state_tracker": StateTracker(story),
        'context': collections.deque(maxlen=CONTEXT_SIZE),
        'replies': collections.deque(maxlen=REPLY_HIST_SIZE)}
    if random.random() > 0.5:
        # we decide to go first
        bot_send_message(tel_obj, update, iface.bot.get_question(sender_id))


def help(tel_obj, update):
    md = mess2dict(update.message)
    help_msg = iface.bot.help(_sender_credentials(md))
    tel_obj.sendMessage(update.message.chat_id, text=help_msg,
                        parse_mode=ParseMode.MARKDOWN)


def echo(tel_obj, update):
    text = update.message.text
    md = mess2dict(update.message)
    sender_fname, sender_lname = _sender_credentials(md)
    logger.info("{} {} says: {}".format(sender_fname, sender_lname, text))

    sender_id = str(md['from']['id'])
    msg_id = str(md["message_id"])

    if text:
        iface.history[sender_id]['context'].append(text)
        rep, _ = self.bot.respond_to(text, sender_id)
        iface.history[sender_id]['replies'].append(rep)
        logger.info('bot replies: {}'.format(rep))
        iface_send_message(tel_obj, update, rep)


def _sender_credentials(mess_dict):
    try:
        sender_fname = md['from']['first_name']
        sender_lname = md['from']['last_name']
    except:
        sender_fname = str(md['from']['id'])
        sender_lname = ""
    return sender_fname, sender_lname


def iface_send_message(tel_obj, update, text):
    tel_obj.sendChatAction(update.message.chat_id, action=ChatAction.TYPING)
    sleep(random.random() * 2 + 1.)
    tel_obj.sendMessage(update.message.chat_id, text=text)


def error(tel_obj, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))


def remove_punktuation(s):
    return ''.join([ch for ch in s if ch not in exclude])


exclude = set(string.punctuation)

if __name__ == "__main__":
    bot = TelegramBot()
    iface = TelegramInterface(bot)
    iface.run()
