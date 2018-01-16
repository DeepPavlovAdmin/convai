# -*- coding: utf-8 -*-

#import pdb
#pdb.set_trace()
import collections
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import string
from ast import literal_eval
from time import sleep
import configparser
from telegram import ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram.parsemode import ParseMode

import sys
sys.path.append(os.getcwd())

from bot_code.story_handler import StoriesHandler  # StoriesHandler (= Passage retrieval)

from bot_code.CC import CC # in bot_code directory
from bot_code.DA import DA_CNN # in bot_code directory
from bot_code.QA import QA, get_opt # in bot_code directory
from bot_code.RULE import RULE
from bot_code.DR import DocumentRetriever




CONFPATH = "config.ini"
conf = configparser.ConfigParser()
if not os.path.exists(CONFPATH):
    print("Creating stub config...\n"
          "You need to replace STUB with your actual token in file {}".format(CONFPATH))
    conf["bot"] = {"TOKEN": "STUB", "CONTEXT_SIZE": 3, "REPLY_HIST_SIZE": 20, "LOGFILE": 'log.txt'}
    with open(CONFPATH, 'wt') as configfile:
        conf.write(configfile)

conf.read(CONFPATH)

TOKEN = conf["bot"]["TOKEN"]
CONTEXT_SIZE = int(conf["bot"]["CONTEXT_SIZE"])
REPLY_HIST_SIZE = int(conf["bot"]["REPLY_HIST_SIZE"])
LOGFILE = conf["bot"]["LOGFILE"]


# Enable logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

class Bot:
    def __init__(self):
        self.history = {}

        self.nTurn = 0
        self.end = False

        self.cuda = False # False by default

        self.updater = Updater(TOKEN) # Team_kAIb_bot (or cnslbot)
        #self.updater = Updater(token="DA008C35-73CD-4A64-8D67-5C922808D6B4", base_url="https://convaibot.herokuapp.com") # convai

        self.name = str(self).split(' ')[-1][:-1]

        self.dp = self.updater.dispatcher

        self.dp.add_handler(CommandHandler("start", start))  # function1 : /start
        self.dp.add_handler(CommandHandler("talk", start1))  # function1 : /talk = /start1
        self.dp.add_handler(CommandHandler("start1", start1))  # function1 : /start1
        self.dp.add_handler(CommandHandler("start2", start2))  # function1 : /start2
        self.dp.add_handler(CommandHandler("start3", start3))  # function1 : /start3
        self.dp.add_handler(CommandHandler("help", help))    # function2 : /help
        self.dp.add_handler(CommandHandler("new", new))      # function3 : /new  : new paragraph
        self.dp.add_handler(CommandHandler("end", end))      # function4 : /end : end conversation & rating

        self.dp.add_handler(MessageHandler([Filters.text], echo))   #

        # Load modules
        QA_mdl_path = '../model/kaib_qa.mdl'
        qa_opt =  get_opt(QA_mdl_path)
        qa_opt['pretrained_model'] = QA_mdl_path
        qa_opt['datatype'] = 'valid'
        #qa_opt['embedding_file'] = '../ParlAI/data/glove.840B.300d.txt'
        qa_opt['embedding_file'] = '' # we don't need it anymore (since all of the embeddings are stored in model file)

        da_checkpoint_dir = "../model/checkpoint_DA/" # bot.sh

        cc_dict_dir = "../model/dict_file_th5.dict"
        cc_checkpoint_dir = "../model/exp-emb300-hs2048-lr0.0001-bs128"

        # DR
        dr_file_path = "../kb/wikipedia_en_all_nopic_2017-08.zim"
        dr_dir_path = "../kb/index"

        #self.DA = DA()
        self.QA = QA(qa_opt, cuda=self.cuda)
        self.DA = DA_CNN(da_checkpoint_dir, cuda=self.cuda)
        self.CC = CC(cc_checkpoint_dir, cc_dict_dir, cuda=self.cuda)
        self.RULE = RULE()
        self.DR = DocumentRetriever(dr_file_path , dr_dir_path)
        #self.corefer = COREFER()

        self.dp.add_error_handler(error)
        self.stories = StoriesHandler()
        logger.info('Start kAIb bot !')

    def power_on(self):
        # Start the Bot
        self.updater.start_polling()

        # Run the bot until the you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.updater.idle()


def mess2dict(mes):
    return literal_eval(str(mes))


def start(bot, update):
    ai.end = False
    md = mess2dict(update.message)
    sender_id = str(md['from']['id'])

    if not (sender_id in ai.history):
        # Currently output is always same whenever we type /start several times

        story = ai.stories.get_one() # get random paragraph from subset of SQuAD paragraph
        #story = "The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederic Buechner. While not Catholic, Buechner has praised writers from Norte Dame and Moreau Seminary created a Buechner Prize for Preaching."

        bot.sendMessage(update.message.chat_id, 'Hello. I am kAIb bot. We are supposed to talk with each other for a while :)')
        bot.sendMessage(update.message.chat_id, 'Here is the sample passage.')
        bot.sendMessage(update.message.chat_id, text=story)
        bot.sendMessage(update.message.chat_id, 'Ask me question about this passage or you can just chitchat with me')
        bot.sendMessage(update.message.chat_id, 'Or you can either type special commands such as /new, /help or /end')

        #ai.history[sender_id] = {"state_tracker": StateTracker(story),'context': collections.deque(maxlen=CONTEXT_SIZE),'replies': collections.deque(maxlen=REPLY_HIST_SIZE)}
        ai.history[sender_id] = {'context': collections.deque(maxlen=CONTEXT_SIZE),'replies': collections.deque(maxlen=REPLY_HIST_SIZE)}
    else :
        story = ai.stories.get_one() # get random paragraph from subset of SQuAD paragraph

        bot.sendMessage(update.message.chat_id, 'Start with new paragraph !')
        bot.sendMessage(update.message.chat_id, 'You can ask me a question about passage, or simple chitchat with me.')
        bot.sendMessage(update.message.chat_id, 'If you want to talk about new passage, please type /new.')

def new(bot, update):
    md = mess2dict(update.message)

    story = ai.stories.get_one() # get random paragraph from subset of SQuAD paragraph
    bot.sendMessage(update.message.chat_id, text=story)

def end(bot, update):
    bot.sendMessage(update.message.chat_id, 'Thanks for chatting with me. Please rate our conversation !')
    bot.sendMessage(update.message.chat_id, 'Quality : How proper my reponse for your query? (0-10)')
    bot.sendMessage(update.message.chat_id, 'Breadth : How diverse topics can I answer? (0-10)')
    bot.sendMessage(update.message.chat_id, 'Engagement : How much did you enjoy chatting with me? (0-10)')
    bot.sendMessage(update.message.chat_id, 'You can type scores and feel free to leave :).  Bye !')
    ai.end = True

def help(bot, update):
    if not ai.end:
        md = mess2dict(update.message)
        sender_id = md['from']['id']
        try:
            sender_fname = md['from']['first_name']
            sender_lname = md['from']['last_name']
        except:
            sender_fname = sender_id
            sender_lname = ''
        help_msg = ("Hello, {} {}!").format(sender_fname, sender_lname)
        bot.sendMessage(update.message.chat_id, text=help_msg, parse_mode=ParseMode.MARKDOWN)
        bot.sendMessage(update.message.chat_id, 'Ask me question about this passage or you can just chitchat with me')
        bot.sendMessage(update.message.chat_id, 'Or you can either type special commands such as /new, /help or /end')
    else:
        bot.sendMessage(update.message.chat_id, 'Currently, our conversation is over :) If you wanna start again, please type /start')


def echo(bot, update):
    # this function is executed when sending message in telegram
    if not ai.end:
        ai.nTurn += 1
        text = update.message.text
        md = mess2dict(update.message)
        try:
            sender_fname = md['from']['first_name']
            sender_lname = md['from']['last_name']
        except:
            sender_fname = str(md['from']['id'])
            sender_lname = ""
        logger.info("{} {} says: {}".format(sender_fname, sender_lname, text))

        sender_id = str(md['from']['id'])
        #msg_id = str(md["message_id"])

        qa_mode = ai.DA.classify_user_query(text, ai.stories.current_story)

        #if random.random() > 0.5:
        if qa_mode:
            da_mode = 'QA'
            #rep_da = 'DA_mode = ' + da_mode
            #bot_send_message(bot, update, rep_da)
            passage2 = ai.DR.retrieve(text)
            #rep_passage = "Secondary passage = \n" + passage2
            #bot_send_message(bot, update, rep_passage)

            augmented_passage = ai.stories.current_story + '\n' + passage2
            rep = ai.QA.get_reply(augmented_passage, text)
        else:
            rep = ai.RULE.get_reply('', '', text) #  seq2seq test
            if(len(rep) > 0): # RULE FIRST
                rep = rep
            else: # IF RULE IS NOT POSSIBLE, USE CC
                rep = ai.CC.get_reply(text, '','')

        logger.info('bot replies: {}'.format(rep))
        bot_send_message(bot, update, rep)

        # History
        ai.history[sender_id]['context'].append(text)
        ai.history[sender_id]['replies'].append(rep)
    else:
        text = update.message.text
        md = mess2dict(update.message)
        try:
            sender_fname = md['from']['first_name']
            sender_lname = md['from']['last_name']
        except:
            sender_fname = str(md['from']['id'])
            sender_lname = ""
        logger.info("{} {} says: {}".format(sender_fname, sender_lname, text))
        logger.info('bot replies: end of conversation')


def bot_send_message(bot, update, text):
    bot.sendChatAction(update.message.chat_id, action=ChatAction.TYPING)
    sleep(random.random() * 2 + 1.)
    bot.sendMessage(update.message.chat_id, text=text)


def error(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))


def remove_punktuation(s):
    return ''.join([ch for ch in s if ch not in exclude])

exclude = set(string.punctuation)

def talk(bot, update):
    ai.end = False
    md = mess2dict(update.message)
    sender_id = str(md['from']['id'])

    # Currently output is always same whenever we type /start several times
    story = "The Berbers along the Barbary Coast (modern day Libya) sent pirates to capture merchant ships and hold the crews for ransom. The U.S. paid protection money until 1801, when President Thomas Jefferson refused to pay and sent in the Navy to challenge the Barbary States, the First Barbary War followed. After the U.S.S. Philadelphia was captured in 1803, Lieutenant Stephen Decatur led a raid which successfully burned the captured ship, preventing Tripoli from using or selling it. In 1805, after William Eaton captured the city of Derna, Tripoli agreed to a peace treaty. The other Barbary states continued to raid U.S. shipping, until the Second Barbary War in 1815 ended the practice."
    ai.stories.current_story = story
    #story = "The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederic Buechner. While not Catholic, Buechner has praised writers from Norte Dame and Moreau Seminary created a Buechner Prize for Preaching."
    bot.sendMessage(update.message.chat_id, text=story)

    ai.history[sender_id] = {'context': collections.deque(maxlen=CONTEXT_SIZE),'replies': collections.deque(maxlen=REPLY_HIST_SIZE)}


def start1(bot, update):
    ai.end = False
    md = mess2dict(update.message)
    sender_id = str(md['from']['id'])

    # Currently output is always same whenever we type /start several times
    story = "The Berbers along the Barbary Coast (modern day Libya) sent pirates to capture merchant ships and hold the crews for ransom. The U.S. paid protection money until 1801, when President Thomas Jefferson refused to pay and sent in the Navy to challenge the Barbary States, the First Barbary War followed. After the U.S.S. Philadelphia was captured in 1803, Lieutenant Stephen Decatur led a raid which successfully burned the captured ship, preventing Tripoli from using or selling it. In 1805, after William Eaton captured the city of Derna, Tripoli agreed to a peace treaty. The other Barbary states continued to raid U.S. shipping, until the Second Barbary War in 1815 ended the practice."
    ai.stories.current_story = story
    #story = "The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederic Buechner. While not Catholic, Buechner has praised writers from Norte Dame and Moreau Seminary created a Buechner Prize for Preaching."
    bot.sendMessage(update.message.chat_id, text=story)

    ai.history[sender_id] = {'context': collections.deque(maxlen=CONTEXT_SIZE),'replies': collections.deque(maxlen=REPLY_HIST_SIZE)}

def start2(bot, update):
    ai.end = False
    md = mess2dict(update.message)
    sender_id = str(md['from']['id'])

    # Currently output is always same whenever we type /start several times
    story = "Many pre-Columbian civilizations established characteristics and hallmarks which included permanent or urban settlements, agriculture, civic and monumental architecture, and complex societal hierarchies. Some of these civilizations had long faded by the time of the first significant European and African arrivals (ca. late 15th–early 16th centuries), and are known only through oral history and through archaeological investigations. Others were contemporary with this period, and are also known from historical accounts of the time. A few, such as the Mayan, Olmec, Mixtec, and Nahua peoples, had their own written records. However, the European colonists of the time worked to eliminate non-Christian beliefs, and Christian pyres destroyed many pre-Columbian written records. Only a few documents remained hidden and survived, leaving contemporary historians with glimpses of ancient culture and knowledge."
    ai.stories.current_story = story
    #story = "The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederic Buechner. While not Catholic, Buechner has praised writers from Norte Dame and Moreau Seminary created a Buechner Prize for Preaching."
    bot.sendMessage(update.message.chat_id, text=story)

    ai.history[sender_id] = {'context': collections.deque(maxlen=CONTEXT_SIZE),'replies': collections.deque(maxlen=REPLY_HIST_SIZE)}

def start3(bot, update):
    ai.end = False
    md = mess2dict(update.message)
    sender_id = str(md['from']['id'])

    # Currently output is always same whenever we type /start several times
    story = "Rajasthan is known for its traditional, colorful art. The block prints, tie and dye prints, Bagaru prints, Sanganer prints, and Zari embroidery are major export products from Rajasthan. Handicraft items like wooden furniture and crafts, carpets, and blue pottery are commonly found here. Shopping reflects the colorful culture, Rajasthani clothes have a lot of mirror work and embroidery. A Rajasthani traditional dress for females comprises an ankle-length skirt and a short top, also known as a lehenga or a chaniya choli. A piece of cloth is used to cover the head, both for protection from heat and maintenance of modesty. Rajasthani dresses are usually designed in bright colors like blue, yellow and orange."
    ai.stories.current_story = story
    #story = "The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederic Buechner. While not Catholic, Buechner has praised writers from Norte Dame and Moreau Seminary created a Buechner Prize for Preaching."
    bot.sendMessage(update.message.chat_id, text=story)

    ai.history[sender_id] = {'context': collections.deque(maxlen=CONTEXT_SIZE),'replies': collections.deque(maxlen=REPLY_HIST_SIZE)}

if __name__ == "__main__":
    ai = Bot()
    ai.power_on()
