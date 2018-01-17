"""
Natural language understanding module with ALICE-bot.
For more information, see: 
"""


from nlu import *


class NLUAlice(NLUBase):
    def __init__(self, config):
        NLUBase.__init__(self, config)

    def get_raw_input(self, input):
        # input is a string, e.g., "Hi there!".
        return input
