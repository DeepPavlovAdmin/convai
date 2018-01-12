import logging
import api_wrappers.convai as convai_api
import requests
import os
import subprocess

from time import sleep
from bot_brain import BotBrain, greet_user
from sys import argv
from config import version, convai_token


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger_bot = logging.getLogger('bot')
bot_file_handler = logging.FileHandler("bot.log")
bot_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
bot_file_handler.setFormatter(bot_log_formatter)
if not logger_bot.handlers:
    logger_bot.addHandler(bot_file_handler)


class DialogTracker:
    def __init__(self, bot_url):
        self._bot = convai_api.ConvApiBot(bot_url)
        self._bot_url = bot_url

        self._chat_fsm = {}
        self._users = {}
        self._text = 'God'
        self._factoid_qas = []

    def start(self):
        while True:
            try:
                res = requests.get(os.path.join(self._bot_url, 'getUpdates'), timeout=5)
                if res.status_code != 200:
                    logger.warn(res.text)
                if len(res.json()) == 0:
                    sleep(0.1)
                    continue

                for m in res.json():
                    logger.info(m)
                    update = convai_api.ConvUpdate(m)
                    if m['message']['text'].startswith('/start '):
                        self._log_user('_start_or_begin_or_test_cmd', update)

                        greet_user(self._bot, update.effective_chat.id)

                        self._text = m['message']['text'][len('/start '):]
                        self._get_qas()
                        self._add_fsm_and_user(update, True)
                        fsm = self._chat_fsm[update.effective_chat.id]
                        fsm.start_convai()
                    elif m['message']['text'] == '/end':
                        self._log_user('_end_cmd', update)
                        fsm = self._chat_fsm[update.effective_chat.id]
                        fsm.return_to_init()
                    elif m['message']['text'] == 'version':
                        self._log_user('version', update)
                        self._add_fsm_and_user(update)
                        fsm = self._chat_fsm[update.effective_chat.id]
                        fsm._send_message("Version is {}".format(version))
                    elif m['message']['text'].startswith('reset'):
                        self._log_user('reset', update)
                        self._add_fsm_and_user(update, True)
                        fsm = self._chat_fsm[update.effective_chat.id]
                        fsm.return_to_init()
                        fsm.return_to_start()
                        fsm._send_message("Hmm....")
                    else:
                        self._log_user('_echo_cmd', update)

                        fsm = self._chat_fsm[update.effective_chat.id]
                        fsm._last_user_message = update.message.text
                        if not fsm._text:
                            fsm._send_message('Text is not given. Please try to type /end and /test to reset the state and get text.')
                            continue

                        # Do we need this if else?
                        if fsm.is_asked():
                            fsm.check_user_answer_on_asked()
                        else:
                            fsm.classify()
            except Exception as e:
                logger.exception(str(e))
            sleep(0.1)

    def _log_user(self, cmd, update):
        logger_bot.info("USER[{}]: {}".format(cmd, update.message.text))

    # TODO: Merge this method with telegram_main
    def _add_fsm_and_user(self, update, hard=False):
        if update.effective_chat.id not in self._chat_fsm:
            fsm = BotBrain(self._bot, update.effective_user, update.effective_chat, self._text_and_qa())
            self._chat_fsm[update.effective_chat.id] = fsm
            self._users[update.effective_user.id] = update.effective_user
        elif update.effective_user.id in self._chat_fsm and hard:
            self._chat_fsm[update.effective_chat.id].clear_all()
            del self._chat_fsm[update.effective_chat.id]
            fsm = BotBrain(self._bot, update.effective_user, update.effective_chat, self._text_and_qa())
            self._chat_fsm[update.effective_chat.id] = fsm
            self._users[update.effective_user.id] = update.effective_user

    def _get_qas(self):
        out = subprocess.check_output(["from_question_generation/get_qnas", self._text])
        questions = [line.split('\t') for line in str(out, "utf-8").split("\n")]
        self._factoid_qas = [{'question': e[0], 'answer': e[1], 'score': e[2]} for e in questions if len(e) == 3]

    def _text_and_qa(self):
        return {'text': self._text, 'qas': self._factoid_qas}


if __name__ == '__main__':
    dt = DialogTracker(convai_token)
    dt.start()
