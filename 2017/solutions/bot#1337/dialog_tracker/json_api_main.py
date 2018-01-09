import api_wrappers.json_wrapper as json_api
import logging
import subprocess

from flask import request, jsonify, url_for, Flask
from config import version
from uuid import uuid4
from bot_brain import BotBrain


app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_bot = logging.getLogger('bot')
bot_file_handler = logging.FileHandler("bot_flask.log")
bot_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
bot_file_handler.setFormatter(bot_log_formatter)
if not logger_bot.handlers:
    logger_bot.addHandler(bot_file_handler)

chat_fsm = {}

@app.route('/start', methods=["POST"])
def start():
    if not request.get_json():
        return jsonify({'message': 'application type should be json'}), 400

    text = request.get_json().get('text', '')

    qas = get_qas(text)
    chat_id = str(uuid4())
    update = json_api.JsonUpdate('', chat_id)
    bot = json_api.JsonApiBot()
    add_fsm_and_user(update, bot, {'text': text, 'qas': qas}, True)
    fsm = chat_fsm[update.effective_chat.id]
    fsm.start_convai()
    log_user('/start', update)

    return jsonify({'chat_id': chat_id})


@app.route('/message', methods=["POST"])
def message():
    if not request.get_json():
        return jsonify({'message': 'application type should be json'}), 400

    chat_id = request.get_json().get('chat_id', None)
    text = request.get_json().get('text', None)
    if chat_id is None or text is None:
        return jsonify({'message': 'Provide chat_id and text'}), 400

    if chat_id not in chat_fsm:
        return jsonify({'message': 'chat_id is not found. Do /start first'}), 404

    update = json_api.JsonUpdate(text, chat_id)
    log_user('/message', update)

    fsm = chat_fsm[update.effective_chat.id]
    fsm._last_user_message = update.message.text
    fsm = chat_fsm[update.effective_chat.id]
    fsm.classify()

    message = fsm._dialog_context[-1][1]

    return jsonify({'text': message})


@app.route('/end', methods=["POST"])
def end():
    if not request.get_json():
        return jsonify({'message': 'application type should be json'}), 400

    chat_id = request.get_json().get('chat_id', None)
    if chat_id is None:
        return jsonify({'message': 'Provide chat_id'}), 400

    if chat_id not in chat_fsm:
        return jsonify({'message': 'chat_id is not found'}), 404

    update = json_api.JsonUpdate('', chat_id)
    log_user('/end', update)

    del chat_fsm[update.effective_chat.id]

    return jsonify({'message': 'ok'})


def add_fsm_and_user(update, bot, text_and_qa, hard=False):
    if update.effective_chat.id not in chat_fsm:
        fsm = BotBrain(bot, update.effective_user, update.effective_chat, text_and_qa)
        chat_fsm[update.effective_chat.id] = fsm
    elif update.effective_user.id in chat_fsm and hard:
        chat_fsm[update.effective_chat.id].reinit_text_based_skills_and_data(text_and_qa)
        chat_fsm[update.effective_chat.id].clear_all()


def get_qas(text):
    out = subprocess.check_output(["from_question_generation/get_qnas", text])
    questions = [line.split('\t') for line in str(out, "utf-8").split("\n")]
    factoid_qas = [{'question': e[0], 'answer': e[1], 'score': e[2]} for e in questions if len(e) == 3]
    return factoid_qas


def log_user(cmd, update):
    logger_bot.info("USER[{}]: {}".format(cmd, update.message.text))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
