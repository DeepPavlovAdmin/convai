from __future__ import print_function

import datetime as dt

import utils
from tools import colors

U = utils.U


def no_op(f):
    f._no_op = True
    return f


class ResponderRole(utils.Singleton):
    name = None
    apply_profanity = True

    def __init__(self, async=False, **kwargs):
        print("Running responder init for class", self.__class__.__name__,
              async, kwargs)
        super(ResponderRole, self).__init__(**kwargs)
        self.async = async

    def new_state(self):
        return None

    def respond_to(
            self, state, last_user_utt_dict, last_bot_utt, user_utt_dict):
        t0 = dt.datetime.now()
        state, ret, score = self._respond_to(
            state, last_user_utt_dict, last_bot_utt, user_utt_dict)
        seconds = (dt.datetime.now() - t0).total_seconds()
        print(colors.colorize('[%s took %.2f s]' % (self.name, seconds),
                              fg='purple'))
        ret = U(ret)
        return state, ret, score

    # Utility function for bots

    def was_i_last(self, last_my_utt, last_bot_utt):
        def norm_utt(s):
            return U(s).lower().strip()
        return (last_bot_utt and last_my_utt and
                norm_utt(last_my_utt) in norm_utt(last_bot_utt))

    # All talkers should override some of the function below:
    @no_op
    def set_article(self, state, article_dict):
        """
        Set the discussed article

        Args:
            state: the talker state
            article_dict: a dict of:
                - text: unicode
                - corefed_text: unicode

        Returns:
            new state
        """
        return state

    def _respond_to(
            self, state, last_user_utt_dict, last_bot_utt, user_utt_dict):
        """
        Compute the new state and response.

        Note: to prevent calling this method decorate _respond_to with @no_op

        Args:
            state: old talker state
            last_user_utt_dict: a dict containing:
                - raw_utt: the raw unicode text
                - spelled_utt: spellchecked, unicode
                - spelled_tags: list of (token, tag)
                - corefed_utt: spelled and coreference resolved, unicode
                - corefed_tags: list of (token, tag)
            last_bot_utt: last bot utterance
            user_utt_dict: current user utterance, in a format identical
                to last_user_utt_dict

        Returns:
            triple containing:
                - new state
                - proposed response
                - response score in range:
                    None, <0, will not be used
                    [0-1]: normal response, will be weighted:
                                    confidence * weight
                    [10-11]: priority response, the score will be
                                    10 + (confidence - 10) * weight
        """
        raise NotImplementedError

    @no_op
    def follow_up(self, state, new_bot_utts):
        """
        Generate a follow-up text
            Args:
            state: old talker state
            new_bot_utts: a sorted by score in decreasing order
                (best utt is first) list of dicts of previous utterances
                with at least the following fields:
                    - utt: the talker's proposed utterance
                    - score: the score (talker confidence * talker weight)
                    - talker_name: the name of the talker class (the talker
                      class itself can be getattred from  module talkers)

        Returns:
            a triple of:
                - state
                - new bot utterance. It can be arbitrary, but typically
                  should contain the top talker's output
                - a confidence score, with semantics:
                    - None or <0: don't show
                    - [0-1]: probablity of selecting the follow-up
                    - >1: always show the follow-up with the highest score
                          (weighted by talker's weight)
        """
        return state, None, None
