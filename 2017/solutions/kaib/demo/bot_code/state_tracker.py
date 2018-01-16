import random
import json
import subprocess


class StateTracker:
    def __init__(self, salt):
        print('initialize state tracker')
        self.salt = salt
        self.get_answer = lambda x: get_answer(self.salt, x)
        #out = subprocess.check_output(["question_generation/get_qnas", self.salt])
        # QG is not work in 970-1, very slow
        out = 'DUMMY QUESTION #1 \n DUMMY QUESTION #2 \n DUMMY QUESTION #3'

        # Original
        #self.questions = [line.split('\t') for line in str(out, "utf-8").split("\n")]

        # Ver1
        #self.questions = []
        #self.questions[0] = 'DUMMY QUESTION #1'
        #self.questions[1] = 'DUMMY QUESTION #2'
        #print('self.questions = ')
        #print(self.questions)
        #self.questions[2] = 'DUMMY QUESTION #3'

        # Ver2
        self.questions = 'DUMMY QUESTION'
        self.used_questions = []
        self._qa_clf = dummy_clf

    def get_question(self):
        print('(state tracker) get question')
        if not self.questions:
            self.questions = self.used_questions
            self.used_questions = []

        index = random.randrange(0, len(self.questions))
        #question = self.questions.pop(index)  # pop returns element from list
        question = 'DUMMY QUESTION'
        self.used_questions.append(question)
        #return question[0] # Original
        return question  # Ken

    def get_reply(self, utterance):
        is_question = self._qa_clf(utterance)
        if is_question:
            return self.get_answer(utterance)
        else:
            return self.get_question()


class StoriesHandler:
    # Retrieve QA passage
    def __init__(self, filename="data/train-v1.1.json"):
        with open(filename) as f:
            dataset = json.load(f)
        self.stories = [par["context"] for text in dataset["data"] for par in text["paragraphs"]]

    def get_one(self):
        self.current_story = random.choice(self.stories)
        return self.current_story


def get_answer(paragraph, question):
    out = subprocess.check_output(["python3", "bi-att-flow/get_answer.py",
                                   "--paragraph", paragraph, "--question", question])
    return str(out, "utf-8")


def get_questions(paragraph):
    return [["STUB QUESTION 1"], ["STUB QUESTION 2"]]


def dummy_clf(string):
    clean = string.strip().lower()
    is_question = False
    if clean[-1] == "?":
        is_question = True
    if clean.split()[0] in ["what", "where", "who", "whom", "when", "how"]:
        is_question = True

    return is_question