import config
import random
import aiml
import os
import glob
import uuid
import math

from collections import defaultdict as dd
from talker.base import ResponderRole


data_path = config.data_alice_path


class AliceState(object):
  # Use a counter in addition to uuids because uuid can sometimes return
  # the same value!!
  num_alices = 0

  __slots__ = ['session_id', 'selected_predicates',
               'my_previous_answer', 'what_i_was_talking']

  def __init__(self):
    AliceState.num_alices += 1
    self.session_id = str(uuid.uuid4()) + str(AliceState.num_alices)
    self.selected_predicates = {}
    self.my_previous_answer = '<no_answer>'
    self.what_i_was_talking = {}


class AliceTalker(ResponderRole):
  name = "alice"
  DISCOUNT = 0.9

  def __init__(self, **kwargs):
    super(AliceTalker, self).__init__()

    aimls = glob.glob(os.path.join(config.alice_path, 'aiml/*.aiml'))
    self.kernel = aiml.Kernel()
    for n in aimls:
      self.kernel.learn(n)
      print "Learned:", n

    self.predicates = {}
    self.i_dont_knows = ["I don't know", "I do not understand"]

    for x in open(os.path.abspath(os.path.dirname(__file__)
        ) + '/alice_predicates.txt'):
      # L = unicode(x, 'utf8').split() #WAS
      L = x.split()
      if len(L) == 0:
        continue
      if x[0] == '#':
        continue
      if x[0] != ' ':
        current = []
        for w in L:
          self.predicates[w] = current
      else:
        current.append(x.strip())

    self.idf = dd(lambda:0)
    for x in open(data_path + '/alice_data/alice_counts.txt'):
      # L = unicode(x, 'utf8').lower().split() #WAS
      L = unicode(x, 'utf8').lower().encode('utf8').split() #IS

      if len(L) == 2:
        self.idf[L[0]] += int(L[1])

    m = 1.0 * sum(self.idf.values())

    for w in self.idf:
      if self.idf[w] > 0:
        self.idf[w] = -math.log(self.idf[w] / m)
    self.max_idf = -math.log(1.0 / m)

  def new_state(self):
    state = AliceState()
    for p in self.predicates:
      state.selected_predicates[p] = random.choice(self.predicates[p])
    return state

  def _respond_to(self, state,
                  last_user_utt_dict, last_bot_utt, user_utt_dict):
    del last_user_utt_dict  # unused
    if self.was_i_last(state.my_previous_answer, last_bot_utt):
      if last_bot_utt not in state.what_i_was_talking:
        state.what_i_was_talking[last_bot_utt] = 0
      state.what_i_was_talking[last_bot_utt] += 1

    question = user_utt_dict['spelled_utt']
    if isinstance(question, unicode):
      question_str = question.encode('utf8')
    else:
      question_str = question

    for pred, value in state.selected_predicates.items():
      self.kernel.setBotPredicate(pred, value)  # type(value) == str

    txt = self.kernel.respond(question_str, state.session_id)
    if len(txt) > 200:
      # due to some AIML errors Kernel produces some times very log texts
      txt = txt[:200] + '...'

    if len(txt.strip()) == 0:  # empty answer
      return state, 'Sometimes it is hard to say something', 0.0

    l_val = self.length_value(txt)
    u_val = self.uniquness_value(question_str, txt)

    val = (0.4 + 0.1 * l_val + 0.5 * u_val) * \
      AliceTalker.DISCOUNT ** state.what_i_was_talking.get(txt, 0)

    for q in self.i_dont_knows:
      if q in txt:
        val /= 10
        break

    txt = unicode(' '.join(txt.split()), 'utf8')
    state.my_previous_answer = txt
    return state, txt, val

  def length_value(self, txt):
    lv = 1.0
    if len(txt) > 100:
      lv -= (len(txt) - 100) / 100
    if lv < 0:
      lv = 0
    return lv

  def uniquness_value(self, question, response):
    "computes the value if the best word introduced by Alice"
    Q = set(question.lower().split()) # intentionally not: word_tokenize
    R = set(response.lower().split())
    new = R - Q

    current_idf = 0.0
    for n in new:
      if not n in self.idf:
        current_idf = self.max_idf
        best_word = n
      else:
        i = self.idf[n]
        if i > current_idf:
          current_idf = i
          best_word = n
    # print 'Most uniq word', n, current_idf / self.max_idf
    return current_idf / self.max_idf


if __name__ == "__main__":
  import sys
  cnt = 0
  prev_txt = ''
  Alicja = AliceTalker()
  state = Alicja.new_state()
  while True:
    state, txt, val = Alicja._respond_to(
        state, "", prev_txt, dict(spelled_utt=unicode(raw_input('Human: '), 'utf8')))
    prev_txt = txt
    print 'Bot: ' + txt, '(value=',str(val) + str(')')
