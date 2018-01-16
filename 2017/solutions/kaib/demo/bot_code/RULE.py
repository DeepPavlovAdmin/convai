import aiml
import re

#from simpler_nlg import SimplerNLG # calee
from nltk.tokenize import casual_tokenize

#import sys

class RULE:
    def __init__(self):
        self.kernel = aiml.Kernel()
        self.kernel.learn("start-rule.xml")
        self.kernel.respond("load aiml b")

        self.num_turn_history = 3


    def get_reply(self, history_context, history_reply, message=""):
        # Note : history_context & history_reply : collections.deque
        history_context_text = ""
        if (len(history_context) >= self.num_turn_history):
            for i in range(self.num_turn_history):
                history_context_text += history_context[len(history_context) + i - self.num_turn_history] + " "

	"""
        history_reply_text = ""
        if (len(history_reply) >= self.num_turn_history):
            for i in range(self.num_turn_history):
                history_reply_text += history_reply[len(history_reply) + i - self.num_turn_history] + " "
	"""

        reply = self.kernel.respond(message)

        return reply


if __name__ == "__main__":
    rule = RULE()
    while True:
        print(rule.get_reply([],[],input('type your mesesage : ')))
    #print(rule.get_reply([], [], "Hi."))
