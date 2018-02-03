from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import collections

from dialog_encdec import DialogEncoderDecoder
from state import prototype_state
import cPickle
import search

MAX_CONTEXT = 3
TOKEN = '390705751:AAHNbIpAsv76PqhRb4pcvmQCWBACej539RE'

TWITTER_MODEL_PREFIX = '/home/ml/mnosew1/SavedModels/Twitter/1489857182.98_TwitterModel'
TWITTER_DICT_FILE = '/home/ml/mnosew1/SavedModels/Twitter/Dataset.dict-5k.pkl'
# TODO: Move these files here.
REDDIT_MODEL_PREFIX = '/home/ml/mnosew1/SavedModels/Reddit/1485212785.88_RedditHRED'
REDDIT_DICT_FILE = '/home/ml/mnosew1/SavedModels/Reddit/Training.dict.pkl'

def help_cmd(bot, update):
    _check_user(update.message.chat_id)
    text = '%s\n%s\n%s %s\n%s' % ('Hello, I am the HRED Bot.',
        'I am trained on data from Twitter and Reddit Politics.',
        'By default, I will talk to you using my Twitter model.',
        'If you would like to switch models, use the /starttwitter or /startreddit commands.',
        'I base my responses on the history of our conversation. If you want to start anew use the /resetcontext command.')
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

def reset_context(bot, update):
    _check_user(update.message.chat_id)
    _reset_user(update.message.chat_id)
    bot.send_message(chat_id=update.message.chat_id, text='Okay, let\'s start over.')

def get_response(bot, update):
    _check_user(update.message.chat_id)
    model_type = ai.history[update.message.chat_id]['model']
    if model_type in ai.models:
        text = ai.models[model_type].get_response(update.message.chat_id, update.message.text)
    else:
        text = 'This model is not implemented yet.'
    bot.send_message(chat_id=update.message.chat_id, text=text)

class Bot(object):

    def __init__(self):
        # For each user, we should keep track of their history as well as which bot they are talking to.
        self.history = {}
        self.updater = Updater(TOKEN)
        self.dp = self.updater.dispatcher

        self._add_cmd_handler('start', help_cmd)
        self._add_cmd_handler('help', help_cmd)
        self._add_cmd_handler('starttwitter', start_twitter)
        self._add_cmd_handler('startreddit', start_reddit)
        self._add_cmd_handler('resetcontext', reset_context)

        msg_handler = MessageHandler(Filters.text, get_response)
        self.dp.add_handler(msg_handler)

        # Set up HRED models.
        self.models = {}
        self.models['twitter'] = HRED_Wrapper(TWITTER_MODEL_PREFIX, TWITTER_DICT_FILE, 'twitter')
        self.models['reddit'] = HRED_Wrapper(REDDIT_MODEL_PREFIX, REDDIT_DICT_FILE, 'reddit')

    def _add_cmd_handler(self, name, fn):
        handler = CommandHandler(name, fn)
        self.dp.add_handler(handler)

    def power_on(self):
        print 'Bot listening...'
        self.updater.start_polling()
        self.updater.idle()

class HRED_Wrapper(object):

    def __init__(self, model_prefix, dict_file, name):
        # Load the HRED model.
        self.name = name
        state_path = '%s_state.pkl' % model_prefix
        model_path = '%s_model.npz' % model_prefix

        state = prototype_state()
        with open(state_path, 'r') as handle:
            state.update(cPickle.load(handle))
        state['dictionary'] = dict_file
        print 'Building %s model...' % name
        self.model = DialogEncoderDecoder(state)
        print 'Building sampler...'
        self.sampler = search.BeamSampler(self.model)
        print 'Loading model...'
        self.model.load(model_path)
        print 'Model built (%s).' % name

        self.speaker_token = '<first_speaker>'
        if name == 'reddit':
            self.speaker_token = '<speaker_1>'

        self.remove_tokens = ['<first_speaker>', '<at>', '<second_speaker>']
        for i in range(0, 10):
            self.remove_tokens.append('<speaker_%d>' % i)

    def _preprocess(self, text):
        text = text.replace("'", " '")
        text = '%s %s </s>' % (self.speaker_token, text.strip().lower())
        return text

    def _format_output(self, text):
        text = text.replace(" '", "'")
        for token in self.remove_tokens:
            text = text.replace(token, '')
        return text

    def get_response(self, user_id, text):
        print '--------------------------------'
        print 'Generating HRED response for user %s.' % user_id
        text = self._preprocess(text)
        ai.history[user_id]['context'].append(text)
        context = list(ai.history[user_id]['context'])
        print 'Using context: %s' % ' '.join(context)
        samples, costs = self.sampler.sample([' '.join(context),], ignore_unk=True, verbose=False, return_words=True)
        response = samples[0][0].replace('@@ ', '').replace('@@', '')
        ai.history[user_id]['context'].append(response)
        response = self._format_output(response)
        print 'Response: %s' % response
        return response

def _reset_user(user_id):
    ai.history[user_id]['context'] = collections.deque(maxlen=MAX_CONTEXT)

def _check_user(user_id):
    # Check if we have the user in our memory, if not... start talking with Twitter bot.
    if user_id not in ai.history:
        ai.history[user_id] = {'model':'twitter',
                                'context': collections.deque(maxlen=MAX_CONTEXT)}

if __name__ == '__main__':
    ai = Bot()
    ai.power_on()
