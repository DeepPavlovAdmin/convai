#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from talker.base import ResponderRole
import time


class ErrorBot(ResponderRole):
    name = "error_bot"

    def set_article(self, state, article_dict):
        print(locals())
        time.sleep(30)
        raise Exception("You only raise once")

    def _respond_to(
            self, state, last_user_utt_dict, last_bot_utt, user_utt_dict):
        print(locals())
        raise Exception("Error bot strikes again")

    def follow_up(self, state, new_bot_utts):
        print(locals())
        return state, new_bot_utts[0]['utt'] + u'followup', 1.0
        raise Exception("Error bot strikes ba-a-a-a-ack")
