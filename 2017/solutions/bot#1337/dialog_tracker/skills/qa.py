import random
import subprocess
from skills.utils import combinate_and_return_answer
# TODO: Remove dependencies on from_* folders;
from from_opennmt_chitchat.get_reply import detokenize, normalize
from fuzzywuzzy import fuzz


class QuestionAskingSkill:
    """Asks factoid question based on text"""
    def __init__(self, qa_skill):
        self._qa_skill = qa_skill

    def predict(self, argument=None):
        return self._qa_skill.ask_question()

    def get_question(self):
        return self._qa_skill._last_factoid_qas.get('question')


class AnswerCheckingSkill:
    """Checks if user answer on asked question was right and responses"""
    def __init__(self, qa_skill):
        self._qa_skill = qa_skill

    def predict(self, user_answer):
        return self._qa_skill.check_user_answer(user_answer)

    def get_answer(self):
        return self._qa_skill._last_factoid_qas.get('answer')


class QuestionAnsweringSkill:
    """Answers on users factoid question"""
    def __init__(self, text):
        self._text = text

    def predict(self, question):
        answer = self._get_answer_to_factoid_question(question)

        msg1 = ["I think that", "It seems that", "I'd like to say that"]
        msg2 = ["correct answer", "answer", "true answer"]
        msg3 = ["is: {}".format(detokenize(normalize(answer))).lower()]
        total_msg = [msg1, msg2, msg3]

        msg = combinate_and_return_answer(total_msg)
        return msg

    def _get_answer_to_factoid_question(self, question):
        # TODO: Remove dependencies on from_* folders;
        out = subprocess.check_output(
            ["python3", "from_factoid_question_answerer/get_answer.py",
             "--paragraph", self._text, "--question", question])
        return str(out, "utf-8").strip()


class QuestionAskingAndAnswerCheckingSkill:
    def __init__(self, qas, user):
        self._user = user
        self._factoid_qas = qas
        self._question_asked = False
        # last asked factoid qas
        self._last_factoid_qas = {}
        self._is_first_incorrect = True

    def ask_question(self):
        if len(self._factoid_qas) == 0:
            return None
        self._question_asked = True
        self._is_first_incorrect = True
        # takes one question from list and removes it
        self._last_factoid_qas = self._factoid_qas[0]
        self._factoid_qas = self._factoid_qas[1:]

        return self._last_factoid_qas['question']

    def _is_user_answer_correct(self, answer):
        true_answer = self._last_factoid_qas['answer']
        # make user answer lowercased + remove ending chars
        true_answer_clean = true_answer.lower().rstrip(' .,;?!')
        user_answer_clean = answer.lower().rstrip(' .,;?!')
        sim = fuzz.ratio(true_answer_clean, user_answer_clean)
        return sim

    def check_user_answer(self, answer):
        if not self._last_factoid_qas:
            return None

        true_answer = self._last_factoid_qas['answer']
        sim = self._is_user_answer_correct(answer)

        if sim > 95:
            msg = "👍"
            if random.random() > 0.6:
                msg1 = ['It is right', 'And its right answer', 'Right']
                msg2 = ['!', ':)']
                msg3 = ["You're smart.", ""]
                msg4 = ["Ask me something or wait for my new question", "Ask me or wait my new question"]
                msg5 = ["🌈", ":)", ""]
                total_msg = [msg1, msg2, msg3, msg4, msg5]
                msg = combinate_and_return_answer(total_msg)
            self._question_asked = False
        elif sim >= 80:
            msg1 = ["I think you mean: {}".format(true_answer), "Did you mean {}?".format(true_answer)]
            msg2 = ["My congratulations", "If you really mean what I think then my congratulations", "Good job"]
            msg3 = ["!", "."]
            msg4 = ["Ask me something or wait for my new question", "Ask me or wait my new question"]
            msg5 = ["🌈", ":)", ""]
            total_msg = [msg1, msg2, msg3, msg4, msg5]
            msg = combinate_and_return_answer(total_msg)
            self._question_asked = False
        else:
            if self._is_first_incorrect is True:

                msg1 = ["You can do better", "Show me your best", "It is incorrect"]
                msg2 = [".", "!", ":)", '¯\_(ツ)_/¯']
                if len(true_answer) > 3:
                    msg3 = ["Hint: first 3 letters is {}.".format(true_answer[:3])]
                else:
                    msg3 = ["Hint: first 2 letters is {}.".format(true_answer[:2])]
                msg4 = ["Try again", "Try again, please"]
                msg5 = ["", "!", "."]
                total_msg = [msg1, msg2, msg3, msg4, msg5]

                msg = combinate_and_return_answer(total_msg)
                self._is_first_incorrect = False
            else:
                msg = "😕"
                if random.random() > 0.5:
                    msg1 = ['Still incorrect', 'Incorrect', 'Maybe other time']
                    msg2 = ['.', ':(']
                    total_msg = [msg1, msg2]
                    msg = combinate_and_return_answer(total_msg)

                msg3 = ['I think that']
                msg4 = ['correct answer', 'true answer', 'answer']
                msg5 = ['is: {}'.format(true_answer)]
                msg6 = [":)", "", "."]
                total_msg = [msg3, msg4, msg5, msg6]
                msg = combinate_and_return_answer(total_msg)

                self._question_asked = False
                self._is_first_incorrect = True
        return msg
