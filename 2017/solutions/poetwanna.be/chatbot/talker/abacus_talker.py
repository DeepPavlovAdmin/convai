#
# Adrian Lancucki 2017
#
from __future__ import division

import re
import numpy as np

from base import ResponderRole


class AbacusTalker(ResponderRole):
    name = 'AbacusTalker'
    # https://stackoverflow.com/questions/5085524/regular-expression-for-simple-arithmetic-string
    # (whitespaces removed)
    expr = r'-?\d+(?:\.\d+(?:E\d+)?)?(([-+/\*]|\*\*)-?\d+(?:\.\d+(?:E\d+)?)?)+'
    is_arithmetic = re.compile(expr)
    inside_parentheses = re.compile(r'\(' + expr + r'\)')
    partial_url = re.compile(r'[a-zA-Z]+\.[a-zA-Z]{2,4}/')
    idk = (None, 'I have no idea.', 0.0)

    def _template_ans(self, s):
        templates = [u"%s.", u"%s.", u"%s.", u"%s?", u"It's %s.",
                     u"I think it's %s.", u"I'm guessing %s.",
                     u"I'm sure it's %s.",
                     u"My robotic senses tell me it's %s."]
        return np.random.choice(templates) % s

    def _seems_to_have_url(self, user_utt):
        if 'http' in user_utt or 'www' in user_utt:
            return True
        return self.partial_url.search(user_utt) is not None

    def _respond_to(
            self, state, last_user_utt_dict, last_bot_utt, user_utt_dict):
        user_utt = user_utt_dict['raw_utt'].lower()
        if self._seems_to_have_url(user_utt):
            return self.idk
        user_utt = user_utt.replace('plus', '+')
        user_utt = user_utt.replace('add', '+')
        user_utt = user_utt.replace('minus', '-')
        user_utt = user_utt.replace('subtract', '-')
        user_utt = user_utt.replace('sub', '-')
        user_utt = user_utt.replace('multiplied by', '*')
        user_utt = user_utt.replace('multipy by', '*')
        user_utt = user_utt.replace('multipy', '*')
        user_utt = user_utt.replace('mult', '*')
        user_utt = user_utt.replace('x', '*')
        user_utt = user_utt.replace('divided by', '/')
        user_utt = user_utt.replace('divide by', '/')
        user_utt = user_utt.replace('div', '/')
        user_utt = user_utt.replace('^', '**')
        user_utt = user_utt.replace('to the power', '**')
        user_utt = user_utt.replace('to a power', '**')
        user_utt = user_utt.replace('to power', '**')
        user_utt = user_utt.replace('pow', '**')
        user_utt = user_utt.replace(' ', '')  # No whitespace

        class SaveResult(object):
            def __init__(self):
                self.val = None
            def __call__(self, val):
                self.val = val
                return val
        save = SaveResult()

        # Iteratively evaluate all parenthesized expressions
        pm = self.inside_parentheses.search(user_utt)
        while pm is not None:
            try:
                user_utt = self.inside_parentheses.sub(
                    lambda m: save(unicode(eval(m.group()))), user_utt)
            except:
                return self.idk
            pm = self.inside_parentheses.search(user_utt)

        # Evaluate first expression without any parentheses
        match = self.is_arithmetic.search(user_utt)
        if match is not None:
            try:
                result = unicode(eval(match.group()))
            except:
                return self.idk
            return (None, self._template_ans(result), 1.0)

        if save.val is not None:
            # We don't match bare numbers, e.g, "I am 15 yo."
            # However, we could have reduced "What is (2+2)" to "What is 4"
            return (None, self._template_ans(save.val), 1.0)

        return self.idk
