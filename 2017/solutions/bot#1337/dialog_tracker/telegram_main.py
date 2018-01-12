import logging
import telegram
import json
import datetime

from random import sample
from config import version, telegram_token
from time import sleep
from bot_brain import BotBrain, greet_user
from sys import argv
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger_bot = logging.getLogger('bot')
bot_file_handler = logging.FileHandler("bot.log")
bot_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
bot_file_handler.setFormatter(bot_log_formatter)
if not logger_bot.handlers:
    logger_bot.addHandler(bot_file_handler)


def load_text_and_qas(filename):
    with open(filename, 'r') as f:
        return json.load(f)


class DialogTracker:
    def __init__(self, token):
        self._bot = telegram.Bot(token)
        self._updater = Updater(bot=self._bot)

        dp = self._updater.dispatcher
        dp.add_handler(CommandHandler("start", self._start_cmd))
        dp.add_handler(CommandHandler("reset", self._reset_cmd))
        dp.add_handler(CommandHandler("stop", self._reset_cmd))
        dp.add_handler(CommandHandler("help", self._help_cmd))
        dp.add_handler(CommandHandler("text", self._text_cmd))
        dp.add_handler(CommandHandler("evaluation_start", self._evaluation_start_cmd))
        dp.add_handler(CommandHandler("evaluation_end", self._evaluation_end_cmd))

        dp.add_handler(MessageHandler(Filters.text, self._echo_cmd))

        dp.add_handler(CallbackQueryHandler(self._eval_keyboard))
        dp.add_error_handler(self._error)

        self._users_fsm = {}
        self._users = {}
        self._text_and_qas = load_text_and_qas('data/squad-25-qas.json')
        self._text_ind = 0
        self._evaluation = {}
        self._eval_suggestions = None

    def start(self):
        self._updater.start_polling()
        self._updater.idle()

    def _reset_cmd(self, bot, update):
        self._log_user('_reset_cmd', update)

        self._add_fsm_and_user(update)
        fsm = self._users_fsm[update.effective_user.id]

        if update.effective_user.id in self._evaluation:
            self._evaluation[update.effective_user.id]['is_running'] = False

        fsm.return_to_init()
        username = self._user_name(update)
        update.message.reply_text(
            "{}, please type /start to begin the journey {}".format(username, telegram.Emoji.MOUNTAIN_RAILWAY)
        )
        update.message.reply_text("Also, you can type /help to get help")

    def _evaluation_start_cmd(self, bot, update):
        self._log_user('_evaluation_start_cmd', update)
        self._evaluation[update.effective_user.id] = {
            'suggestions': None,
            'thread': [],
            'is_running': True,
            'text': self._text()
        }

        update.message.reply_text("Evaluation mode is on!")

    def _evaluation_end_cmd(self, bot, update):
        self._log_user('_evaluation_end_cmd', update)
        self._evaluation[update.effective_user.id]['is_running'] = False
        update.message.reply_text("Evaluation mode is off!")
        keyboard = [
            [
                telegram.InlineKeyboardButton("Bad", callback_data='overall_bad'),
                telegram.InlineKeyboardButton("Neutral", callback_data='overall_neutral'),
                telegram.InlineKeyboardButton("Good", callback_data='overall_good')
            ],
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        update.message.reply_text('"Evaluate overall dialogue quality, please: "', reply_markup=reply_markup)

    def _start_cmd(self, bot, update):
        self._log_user('_start_cmd', update)

        self._text_ind += 1

        logger_bot.info("BOT[_start_cmd] text_id: {}".format(self._text_ind))

        update.message.reply_text("\"{}\"".format(self._text()))

        greet_user(bot, update.effective_chat.id)

        self._add_fsm_and_user(update, True)
        fsm = self._users_fsm[update.effective_user.id]
        fsm.start()

    def _help_cmd(self, bot, update):
        self._log_user('_help_cmd', update)

        self._add_fsm_and_user(update)

        message = ("/start - starts the chat\n"
                   "/text - shows current text to discuss\n"
                   "/help - shows this message\n"
                   "/reset - reset the bot\n"
                   "/stop - stop the bot\n"
                   "/evaluation_start - start the evaluation mode \n"
                   "/evaluation_end - end the evaluation mode and save eval data \n"
                   "\n"
                   "Version: {}".format(version))
        update.message.reply_text(message)

    def _text_cmd(self, bot, update):
        self._log_user('_text_cmd', update)

        self._add_fsm_and_user(update)
        update.message.reply_text("The text: \"{}\"".format(self._text()))

    def _make_suggest_dict(self, eval_suggestions):
        suggest_dict = {}
        for klass, suggests in eval_suggestions:
            if suggests[0] is not None:
                for ind, suggest in enumerate(suggests):
                    label = klass + " | " + str(suggest)
                    key = "{}-{}".format(klass, ind)
                    suggest_dict[key] = suggest
            else:
                suggest_dict[klass] = klass
        suggest_dict["OTHER"] = "OTHER"
        return suggest_dict

    def _check_choosed(self, eval_suggestions, choosed_by_user):
        suggest_dict = self._make_suggest_dict(eval_suggestions)
        if choosed_by_user not in suggest_dict.keys():
            return False
        else:
            return suggest_dict[choosed_by_user]

    def _echo_cmd(self, bot, update):
        self._log_user('_echo_cmd', update)

        self._add_fsm_and_user(update)

        username = self._user_name(update)
        fsm = self._users_fsm[update.effective_user.id]
        fsm._last_user_message = update.message.text
        user_id = update.effective_user.id

        # Evaluation choice checking
        if user_id in self._evaluation and self._evaluation[user_id]['is_running'] and self._evaluation[user_id]['suggestions']:
            suggestions = self._evaluation[user_id]['suggestions']
            last_user_message, bot_msg = fsm._dialog_context[-1]
            choosed_by_user = update.message.text
            choosed_val = self._check_choosed(suggestions, choosed_by_user)
            if choosed_val:
                update.message.reply_text('Thanks!')
                self._evaluation[user_id]['suggestions'] = None
                row_user = {
                    'text': last_user_message,
                    'userId': 'Human'
                }
                row_bot = {
                    'text': bot_msg,
                    'choosed_by_user': {'class': choosed_by_user, 'sentence_data': choosed_val},
                    'evaluation': '2', # 3 ili 2?
                    'userId': 'Bot'
                }
                self._evaluation[user_id]['thread'].append(row_user)
                self._evaluation[user_id]['thread'].append(row_bot)
            else:
                update.message.reply_text('Your choice is incorrect. Please, try again')
            return

        if fsm.is_init():
            update.message.reply_text(
                "{}, please type /start to begin the journey {}.".format(username, telegram.Emoji.MOUNTAIN_RAILWAY)
            )
            update.message.reply_text("Also, you can type /help to get help")
        else:
            fsm.classify()
            if user_id in self._evaluation and self._evaluation[user_id]['is_running']:
                keyboard = [
                    [telegram.InlineKeyboardButton("Good", callback_data='eval_good')],
                    [telegram.InlineKeyboardButton("Bad", callback_data='eval_bad')  ]
                ]
                reply_markup = telegram.InlineKeyboardMarkup(keyboard)
                update.message.reply_text('Please choose:', reply_markup=reply_markup)

    def _make_suggestions_keyboard(self, suggestions):
        keyboard = []
        suggest_dict = self._make_suggest_dict(suggestions)
        return "\n".join(["{} <> {}".format(k, v) for k, v in suggest_dict.items()])

    def _eval_keyboard(self, bot, update):
        query = update.callback_query
        logger_bot.info("USER[_button]: {}".format(query.data))
        user_id = update.effective_user.id
        fsm = self._users_fsm[user_id]
        last_user_message, bot_msg = fsm._dialog_context[-1]
        if query.data == 'eval_good':
            bot.edit_message_text(
                text="Thanks! Continue to chat, please",
                chat_id=query.message.chat_id,
                message_id=query.message.message_id
            )
            self._evaluation[user_id]['suggestions'] = None
            row_user = {
                'text': last_user_message,
                'userId': 'Human'
            }
            row_bot = {
                'text': bot_msg,
                'evaluation': '3',
                'userId': 'Bot'
            }
            self._evaluation[user_id]['thread'].append(row_user)
            self._evaluation[user_id]['thread'].append(row_bot)
        elif query.data == 'eval_bad':
            suggestions = fsm.generate_suggestions()
            self._evaluation[user_id]['suggestions'] = suggestions
            reply_markup = self._make_suggestions_keyboard(suggestions)
            bot.edit_message_text('Please choose: \n {}'.format(reply_markup),
                chat_id=query.message.chat_id, message_id=query.message.message_id)
        elif 'overall_' in query.data:
            self._evaluation[user_id]['evaluation'] = {'userId': 'Human', 'quality': query.data}
            bot.edit_message_text('Evaluation is completed. Thanks!',
                chat_id=query.message.chat_id, message_id=query.message.message_id
            )
            filename = '/src/data/evaluations/{}-{}.json'.format(datetime.datetime.now(), user_id)
            with open(filename, 'w') as f:
                json.dump(self._evaluation[user_id], f, indent=4, sort_keys=True)


    def _log_user(self, cmd, update):
        logger_bot.info("USER[{}]: {}".format(cmd, update.message.text))

    def _create_fsm(self, update):
        fsm = BotBrain(self._bot, update.effective_user, update.effective_chat, self._text_and_qa())
        self._users_fsm[update.effective_user.id] = fsm
        self._users[update.effective_user.id] = update.effective_user

    def _add_fsm_and_user(self, update, hard=False):
        if update.effective_user.id not in self._users_fsm:
            self._create_fsm(update)
        elif update.effective_user.id in self._users_fsm and hard:
            self._users_fsm[update.effective_user.id].clear_all()
            del self._users_fsm[update.effective_user.id]
            self._create_fsm(update)

    def _error(self, bot, update, error):
        logger.warn('Update "%s" caused error "%s"' % (update, error))

    def _user_name(self, update):
        return self._users[update.effective_user.id].first_name

    def _text(self):
        return self._text_and_qa()['text']

    def _text_and_qa(self):
        return self._text_and_qas[self._text_ind % len(self._text_and_qas)]


if __name__ == '__main__':
    token = telegram_token
    dt = DialogTracker(token)
    dt.start()
