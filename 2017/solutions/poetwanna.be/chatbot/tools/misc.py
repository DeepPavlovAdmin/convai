#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import config

Q_WORDS = set("what where when who how why which whose what's where's when's "
              "who's how's".split())


def is_question(text):
    tokens = text.split()
    if not tokens:
        return False
    return tokens[-1][-1] == '?' or Q_WORDS.intersection(set(tokens))


def question_bonus(text):
    if is_question(text.lower()):
        print("Giving bonus to ", text)
        return config.questionmark_bonus
    else:
        return 0.0
