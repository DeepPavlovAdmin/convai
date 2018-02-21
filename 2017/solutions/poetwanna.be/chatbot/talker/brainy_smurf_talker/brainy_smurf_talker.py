import config
import random

from collections import defaultdict as dd
from nltk.tokenize import word_tokenize

from talker.base import ResponderRole
from tools import profanity, tokenizer
from tools.simple_wikipedia.title_finder import TitleFinder
from utils import U

data_path = config.swt_data_path


def max_t(t, x):
  if x <= t:
    return x
  return t

digits = set("0123456789")

def lower(s):
  return U(s).lower()

def brainy_smurf_intro():
  return random.choice([
     'Fun fact:',
     'Well...',
     'Did you know that?',
     "I've read that",
     "World is strange...",
     "Isn't it interesting?",
     "I have to say this!",
     "It is worth to mention:",
     "Interesting fact:",
     "I can't imagine that!",
     "How about that?"
  ])

brainy_trigger_word = ["information", "article", "text", "paragraph",
                       "passage"] #TODO: what else?

def trigger(question):
  return any(w in question.lower() for w in brainy_trigger_word)

def document_score(title, article, search_score, position, simple):
  if simple: #TODO: title in
    ss_score = min(1, search_score * search_score / 2.0)
    val = 0.3 * ss_score
    if unicode(title, 'utf8').lower() in article:
      val += 0.4
    if set(title) <= digits:
      val /= 5
    if position == 0:
      val += 0.3
    elif position == 1:
      val += 0.15
    elif position == 2:
      val += 0.05
    return val
  return 1.0

def sentence_length_score(sentence):
  length = len(sentence) # with a penalty for longer utf8 codes

  if 20 <= length <= 60:
    return 1.0

  if length < 10:
    return 0

  if length < 20:
    return length / 21.0

  return max(0, 1 - 0.01 * (length - 60))

def sentence_content_score(sentence):
  r = 0.0
  if 'e.g.' in sentence[-4:]:
    return 0.0

  if not digits & set(sentence):
    r += 0.1
  try:
    unicode(sentence)
    r += 0.5
  except:
    pass
  if not ' this ' in ' ' + sentence.lower():
    r += 0.4
  return r


#######################################

class BrainySmurfState(object):
  __slots__ = ['position', 'sentences', 'my_previous_answer', 'rounds',
               'my_last_score', 'my_last_text']
  def __init__(self):
    self.position = 0
    self.sentences = []
    self.my_previous_answer = '<no_answer>'
    self.rounds = 0
    self.my_last_text = ""
    self.my_last_score = 0


class BrainySmurfTalker(ResponderRole):
  name = "brainy_smurf"

  def __init__(self, **kwargs):
    super(BrainySmurfTalker, self).__init__()
    self.TA = TitleFinder()

    sentences = {}
    for x in open(data_path + 'interesting_sentences.txt'):
      L = x.split()
      if len(L) == 0:
        continue
      if L[0] == 'TITLE:':
        title = ' '.join(L[1:])
        sentences[title] = []
      else:
        sentences[title].append(x.strip())
    self.sentences = sentences

    self.useful_titles = {}

    for t in sentences:
      ut = unicode(t, 'utf8')
      key = ' '.join(word_tokenize(ut)).lower().encode('utf8')

      self.useful_titles[key] = t

  def find_phrases(self, doc):
    result = set()
    for length in range(1,5):
      for start in range(len(doc) + length - 1):
        key = ' '.join(doc[start:start+length])
        if key in self.useful_titles:
          result.add(self.useful_titles[key])
    return result

  def new_state(self):
      return BrainySmurfState()

  def sentence_score(self, doc_zero, sentence, title):
    doc_zero = set(map(lower, doc_zero))
    title    = map(lower, word_tokenize(title))
    cnt = 0.0
    mx = 0.0
    ts = {}

    for w in title:
      idf = self.TA.get_idf(w)

      if w in doc_zero:
        cnt += idf
      mx += idf

    sc = 0.75 * cnt / mx

    sc += 0.15 * sentence_length_score(sentence)
    sc += 0.1 * sentence_content_score(sentence)

    return sc


  def set_article(self, state, article_dict):
    article = U(article_dict['text'])
    doc_zero = word_tokenize(article)
    article_lower = article.lower()

    doc_zero = [w.lower().encode('utf8') for w in doc_zero]
    titles = self.find_phrases(doc_zero)

    length_for_phrases = {}

    for t in titles:
      L = t.split()
      if len(L) > 1:
        print 'FOUND PHRASE: ', t
        length_for_phrases[t] = len(L)
      else:
        idf = self.TA.get_idf(t.lower())
        if idf > 7.5:
          print 'FOUND PHRASE: ', t, idf
          length_for_phrases[t] = 1
    print

    term_score = self.TA.idf_term_score(doc_zero)

    new_query = []

    for i, (score,term) in enumerate(term_score):
      if i <= 5 or score >= 12.0 and not set(term) <= digits:
        new_query.append(term)

    print 'QUERYING SIMPLE-WIKI:', new_query

    res = self.TA.find_document_zero(new_query).items()

    S = []

    position = 0
    score_for_title = dd(lambda:0)

    for t,search_score in sorted(res, key = lambda x : -x[1])[:12]:
      score_for_title[t] = 0.7 * search_score

    for t,length in length_for_phrases.items():
      if length == 1:
        sc = 0.3
      elif length == 2:
        sc = 0.7
      elif length == 3:
        sc = 1.0
      else:
        sc = 1.4
      score_for_title[t] += 0.6 * sc

    for t,search_score in score_for_title.items():
      print 'TITLE:',search_score, t, t in self.sentences
      if t in self.sentences:
        first = True
        for s in self.sentences[t]:
          A = document_score(t, article_lower, search_score, position, True)
          B = self.sentence_score(doc_zero, s, t)
          if first and t in length_for_phrases:
            B += 0.2
          S.append( ((1.5 * A + B)/2.5, s) )
          #print '   ', s
          first = False
      position += 1

    S.sort()
    S.reverse()

    print
    for v,s in S[:3]:
      print 'BRAINY SENTENCE:', s, v
    print 'TOTAL NUMBER OF BRAINY SENTENCES:', len(S)
    print
    state.sentences = S

    return state

  def _respond_to(self, state,
                  last_user_utt_dict, last_bot_utt, user_utt_dict):
    del last_user_utt_dict  # unused
    state.rounds += 1

    question = user_utt_dict['corefed_utt']
    if state.sentences == None:
      return state, 'Adrian, give me an article!', 0.0

    if self.was_i_last(state.my_previous_answer, last_bot_utt):
      state.position += 1

    if state.position >= len(state.sentences):
      return state, 'I have no more to say. Maybe we should finish now?', 0.01

    val, txt = state.sentences[state.position]

    val = 0.9 * val + 0.1 * trigger(question)
    val *= 0.95 ** state.position # discount factor

    state.my_previous_answer = txt

    # Look for curse words
    if [t for t in tokenizer.tokenize(txt.lower(), correct_spelling=False)
        if t in profanity.badwords]:
      val = 0.0

    response = U(brainy_smurf_intro() + ' ' + txt)
    state.my_last_text = response
    state.my_last_score = val

    return state, response, val

  def follow_up(self, state, new_bot_utts):
    if new_bot_utts[0]['talker_name'] == self.__class__.__name__:
      return (state, None, None)
    new_bot_utt = new_bot_utts[0]['utt']
    # don't followup long utterances and don't start early
    if len(new_bot_utt) > 40 or state.rounds < 3:
      return state, None, None
    fup_prob = (0.45 * state.my_last_score) ** 1.5
    return state, new_bot_utt + ' ' + state.my_last_text, fup_prob

if __name__ == "__main__":
  cnt = 0
  Brainy = BrainySmurfTalker()
  state = Brainy.new_state()
  for x in open('/data/sample_paragraphs/moscow_paragraphs.txt'):
    if random.randint(0,30) != 0:
      continue
    x = x.strip()

    print 'DOCUMENT'
    print x
    print '======================================'
    Brainy.set_article( state, {'text' : unicode(x, 'utf8')})
    print
    print
    cnt += 1
    if cnt > 1:
      break

