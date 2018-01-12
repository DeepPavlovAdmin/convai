import json
import logging
import requests
import os
from uuid import uuid4
from from_opennmt_chitchat.get_reply import normalize, detokenize

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class JsonApiBot:

    def send_message(self, chat_id, text, reply_markup=None):
        text = detokenize(text)
        message = {'chat_id': chat_id, 'text': text}
        logger.info("JsonApiBot#send_message: {}".format(text))
        return message


class JsonUpdate:
    def __init__(self, text, chat_id):
        text = text.replace('"', " ").replace("`", " ").replace("'", " ")
        self.effective_chat = JsonChat(chat_id)
        self.message = JsonMessage(text)
        self.effective_user = JsonUser()


class JsonChat:
    def __init__(self, id_):
        self.id = id_


class JsonMessage:
    def __init__(self, text):
        self.text = text


class JsonUser:
    def __init__(self):
        self.first_name = 'Anonym'
        self.id = str(uuid4())
