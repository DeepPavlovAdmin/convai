import json
import logging
import requests
import os
from uuid import uuid4
from from_opennmt_chitchat.get_reply import normalize, detokenize

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvApiBot:
    def __init__(self, bot_url):
        self._bot_url = bot_url

    def send_message(self, chat_id, text, reply_markup=None):
        text = detokenize(text)
        data = {'text': text, 'evaluation': 0}
        message = {'chat_id': chat_id, 'text': json.dumps(data)}
        logger.info("ConvApiBot#send_message: {}".format(text))

        res = requests.post(
            os.path.join(self._bot_url, 'sendMessage'),
            json=message,
            headers={'Content-Type': 'application/json'}
        )
        if res.status_code != 200:
            logger.warn(res.text)


class ConvUpdate:
    def __init__(self, message):
        text = message['message']['text'].replace('"', " ").replace("`", " ").replace("'", " ")
        self.effective_chat = ConvChat(message['message']['chat']['id'])
        self.message = ConvMessage(text)
        self.effective_user = ConvUser()


class ConvChat:
    def __init__(self, id_):
        self.id = id_


class ConvMessage:
    def __init__(self, text):
        self.text = text


class ConvUser:
    def __init__(self):
        self.first_name = 'Anonym'
        self.id = str(uuid4())
