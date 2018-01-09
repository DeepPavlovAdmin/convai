import logging
import random
import subprocess
import threading

import config
import requests
import skills.qa as qa
import skills.chitchat as chitchat
import skills.summary as summary
import skills.topic as topic

from skills.utils import combinate_and_return_answer

# TODO: Remove dependencies on from_* folders;
from from_opennmt_chitchat.get_reply import normalize, detokenize
from transitions.extensions import LockedMachine as Machine

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger_bot = logging.getLogger('bot')
bot_file_handler = logging.FileHandler("bot.log")
bot_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
bot_file_handler.setFormatter(bot_log_formatter)
if not logger_bot.handlers:
    logger_bot.addHandler(bot_file_handler)


# TODO: Move to skills
def greet_user(bot, chat_id):
    hello_messages_1 = ['Well hello there!', 'How’s it going?', 'What’s up?',
                        'Yo!', 'Alright mate?', 'Whazzup?', 'Hiya!',
                        'Nice to see you!', 'Good to see you!']
    hello_messages_2 = [""]

    greet_messages = [hello_messages_1, hello_messages_2]
    msg = combinate_and_return_answer(greet_messages)
    bot.send_message(chat_id=chat_id, text=msg)


class BotBrain:
    """Main class that controls dialog flow"""

    states = ['init', 'started', 'waiting', 'classifying']
    wait_messages = [
        "What do you feel about the text?", "Do you like this text?",
        "Do you know familiar texts?", "Can you write similar text?",
        "Do you like to chat with me?", "Are you a scientist?",
        "What do you think about ConvAI competition?",
        "Do you like to be an assessor?",
        "What is your job?"
    ]

    # TODO: move to config file?
    CHITCHAT_URL = 'tcp://opennmtchitchat:5556'
    FB_CHITCHAT_URL = 'tcp://opennmtfbpost:5556'
    SUMMARIZER_URL = 'tcp://opennmtsummary:5556'
    BIGARTM_URL = 'http://bigartm:3000'
    ALICE_URL = 'http://alice:3000'
    INTENT_URL = 'http://intent_classifier:3000/get_intent'

    CLASSIFY_ANSWER = 'ca'
    CLASSIFY_QUESTION = 'cq'
    CLASSIFY_REPLICA = 'cr'
    CLASSIFY_FB = 'cf'
    CLASSIFY_ASK_QUESTION = 'caq'
    CLASSIFY_ALICE = "calice"
    CLASSIFY_SUMMARY = "csummary"
    CLASSIFY_TOPIC = "ctopic"

    # TODO: move to config file?
    MESSAGE_CLASSIFIER_MODEL = "model_all_labels.ftz"

    ASK_QUESTION_ON_WAIT_PROB = 0.5
    MAX_WAIT_TURNS = 4

    def __init__(self, bot, user=None, chat=None, text_and_qa=None):
        self.machine = Machine(model=self, states=BotBrain.states, initial='init')

        # Start part
        self.machine.add_transition('start', 'init', 'started', after='after_start')
        self.machine.add_transition('start_convai', 'init', 'started', after='after_start')

        # Universal states
        self.machine.add_transition('return_to_start', '*', 'started', after='after_start')
        self.machine.add_transition('return_to_wait', '*', 'waiting', after='after_wait')
        self.machine.add_transition('return_to_init', '*', 'init', after='clear_all')

        # Classify user utterance part
        self.machine.add_transition('classify', '*', 'classifying', after='get_class_of_user_message')

        # Too long wait part
        self.machine.add_transition('user_off', 'waiting', 'init', after='propose_conversation_ending')

        self._bot = bot
        self._user = user
        self._chat = chat
        self._text_and_qa = text_and_qa
        self._too_long_waiting_cntr = 0
        self._last_user_message = None
        self._threads = []
        self.reinit_text_based_skills_and_data(text_and_qa)
        self._init_stateless_skills()
        self._dialog_context = []

    def _init_stateless_skills(self):
        self._opensub_chitchat_skill = chitchat.OpenSubtitlesChitChatSkill(BotBrain.CHITCHAT_URL)
        self._fb_chitchat_skill = chitchat.FbChitChatSkill(BotBrain.FB_CHITCHAT_URL)
        self._alice_chitchat_skill = chitchat.AliceChitChatSkill(BotBrain.ALICE_URL)

    def _skill_exec_wrap(self, skill, *args):
        result = skill.predict(*args)
        if result:
            self._send_message(self._filter_seq2seq_output(result))
        else:
            result = self._alice_chitchat_skill.predict(self._last_user_message, self._dialog_context)
            if result:
                self._send_message(self._filter_seq2seq_output(result))
            else:
                self._send_message(random.sample(BotBrain.wait_messages, 1)[0])

        self.return_to_wait()

    def reinit_text_based_skills_and_data(self, text_and_qa):
        self._text_and_qa = text_and_qa
        self._text = self._text_and_qa['text']
        qa_skill = qa.QuestionAskingAndAnswerCheckingSkill(self._text_and_qa['qas'], self._user)
        self._question_ask_skill = qa.QuestionAskingSkill(qa_skill)
        self._answer_check_skill = qa.AnswerCheckingSkill(qa_skill)
        self._question_answerer_skill = qa.QuestionAnsweringSkill(self._text)
        self._summarization_skill = summary.SummarizationSkill(self.SUMMARIZER_URL, self._text)
        self._topic_skill = topic.TopicDetectionSkill(self.BIGARTM_URL, self._text)

    def after_start(self):
        self._cancel_timer_threads(presereve_cntr=False)

        def _say_about_topic_if_user_inactive():
            if self.is_started():
                self._skill_exec_wrap(self._topic_skill)

        def _ask_question_if_user_inactive():
            if self.is_started():
                self._skill_exec_wrap(self._question_ask_skill)

        if random.random() > 0.5:
            t = threading.Timer(config.WAIT_TIME, _ask_question_if_user_inactive)
        else:
            t = threading.Timer(config.WAIT_TIME, _say_about_topic_if_user_inactive)
        t.start()
        self._threads.append(t)

    # Debug and human evaluation function (/evaluation_start mode in Telegram)
    # Should be moved somewhere else
    def generate_suggestions(self):
        def process_tsv(tsv):
            payload = []

            for line in tsv.split('\n'):
                _, resp, score = line.split('\t')
                score = float(score)
                payload.append((resp, score))
            payload = sorted(payload, key=lambda x: x[1], reverse=True)[:3]
            return payload

        answer = self._answer_check_skill.get_answer()
        question = self._question_ask_skill.get_question()

        class_to_string = {
            BotBrain.CLASSIFY_ASK_QUESTION: 'Factoid question',
            BotBrain.CLASSIFY_ANSWER: 'Answer to Factoid question',
            BotBrain.CLASSIFY_QUESTION: 'Factoid question from user',
            BotBrain.CLASSIFY_FB: 'Facebook seq2seq',
            BotBrain.CLASSIFY_REPLICA: 'OpenSubtitles seq2seq',
            BotBrain.CLASSIFY_ALICE: 'Alice',
            BotBrain.CLASSIFY_SUMMARY: 'Summary'
        }

        raw_fb_response = self._fb_chitchat_skill._get_opennmt_fb_reply(
            self._last_user_message, self._dialog_context, self._text, False
        )
        raw_opensub_response = self._opensub_chitchat_skill._get_opennmt_chitchat_reply(
            self._last_user_message, self._dialog_context, False
        )
        fb_replicas = process_tsv(raw_fb_response)
        opensubtitle_replicas = process_tsv(raw_opensub_response)
        alice_replicas = [self._alice_chitchat_skill.predict(self._last_user_message, self._dialog_context)]
        summaries = self._summarization_skill._get_summaries(False)

        result = [
            (class_to_string[BotBrain.CLASSIFY_ASK_QUESTION], [question]),
            (class_to_string[BotBrain.CLASSIFY_ANSWER], [answer]),
            (class_to_string[BotBrain.CLASSIFY_QUESTION], [None]),
            (class_to_string[BotBrain.CLASSIFY_FB], fb_replicas),
            (class_to_string[BotBrain.CLASSIFY_REPLICA], opensubtitle_replicas),
            (class_to_string[BotBrain.CLASSIFY_ALICE], alice_replicas),
            (class_to_string[BotBrain.CLASSIFY_SUMMARY], [summaries]),
            ('Topic Modelling', [self._topic_skill.predict()])
        ]
        return result

    def after_wait(self):
        self._cancel_timer_threads(presereve_cntr=True)

        def _too_long_waiting_if_user_inactive():
            if self.is_waiting() and self._too_long_waiting_cntr < BotBrain.MAX_WAIT_TURNS:
                if random.random() > BotBrain.ASK_QUESTION_ON_WAIT_PROB:
                    self._skill_exec_wrap(self._question_ask_skill)
                else:
                    self._send_message(random.sample(BotBrain.wait_messages, 1)[0])
                self.return_to_wait()
            elif self.is_waiting() and self._too_long_waiting_cntr > BotBrain.MAX_WAIT_TURNS:
                self.user_off()
                self._too_long_waiting_cntr = 0
            else:
                self._too_long_waiting_cntr = 0

        self._too_long_waiting_cntr += 1

        t = threading.Timer(config.WAIT_TOO_LONG, _too_long_waiting_if_user_inactive)
        t.start()
        self._threads.append(t)

    def propose_conversation_ending(self):
        self._cancel_timer_threads()

        self._send_message(("Seems you went to the real life."
                            "Type /start to replay."))

    def clear_all(self):
        self._cancel_timer_threads()

    def get_class_of_user_message(self):
        self._cancel_timer_threads()

        message_class = self._classify(self._last_user_message)
        self._last_classify_label = message_class
        self._classify_user_utterance(message_class)

    def _classify(self, text):
        text = normalize(text)
        cmd = "echo \"{}\" | /fasttext/fasttext predict /src/data/{} -".format(text, BotBrain.MESSAGE_CLASSIFIER_MODEL)
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        res = str(output, "utf-8").strip()
        logger.info(res)

        intent = self._get_intent(text)
        if intent is not None:
            return intent

        if res == '__label__0':
            return BotBrain.CLASSIFY_REPLICA
        elif res == '__label__1':
            return BotBrain.CLASSIFY_QUESTION
        elif res == '__label__2':
            return BotBrain.CLASSIFY_FB
        elif res == '__label__3':
            return BotBrain.CLASSIFY_ANSWER
        elif res == '__label__4':
            return BotBrain.CLASSIFY_ALICE

    def _classify_user_utterance(self, clf_type):
        self._cancel_timer_threads()

        if clf_type == BotBrain.CLASSIFY_ANSWER:
            self._skill_exec_wrap(self._answer_check_skill, self._last_user_message)
        elif clf_type == BotBrain.CLASSIFY_QUESTION:
            self._skill_exec_wrap(self._question_answerer_skill, self._last_user_message)
        elif clf_type == BotBrain.CLASSIFY_REPLICA:
            self._skill_exec_wrap(self._opensub_chitchat_skill, self._last_user_message, self._dialog_context)
        elif clf_type == BotBrain.CLASSIFY_FB:
            self._skill_exec_wrap(self._fb_chitchat_skill, self._last_user_message, self._dialog_context, self._text)
        elif clf_type == BotBrain.CLASSIFY_ASK_QUESTION:
            self._skill_exec_wrap(self._question_ask_skill)
        elif clf_type == BotBrain.CLASSIFY_ALICE:
            self._skill_exec_wrap(self._alice_chitchat_skill, self._last_user_message, self._dialog_context)
        elif clf_type == BotBrain.CLASSIFY_SUMMARY:
            self._skill_exec_wrap(self._summarization_skill)
        elif clf_type == BotBrain.CLASSIFY_TOPIC:
            self._skill_exec_wrap(self._topic_skill)

    def _get_intent(self, text):
        r = requests.post(self.INTENT_URL, json={'text': text})
        intent = r.json()['intent']
        score = r.json()['score']

        if score and score > 0.9:
            return intent
        return None

    def _send_message(self, text, reply_markup=None):
        text = text.strip()
        logger_bot.info("BOT[_send_message]: {}".format(text))

        self._bot.send_message(
            chat_id=self._chat.id,
            text=text,
            reply_markup=reply_markup
        )
        if self._last_user_message is None:
            self._last_user_message = ""
        text = text.replace('"', " ").replace("`", " ").replace("'", " ")
        self._dialog_context.append((self._last_user_message, text))

    def _cancel_timer_threads(self, presereve_cntr=False):
        if not presereve_cntr:
            self._too_long_waiting_cntr = 0

        [t.cancel() for t in self._threads]

    def _filter_seq2seq_output(self, s):
        s = normalize(str(s))
        s = detokenize(s)
        return s
