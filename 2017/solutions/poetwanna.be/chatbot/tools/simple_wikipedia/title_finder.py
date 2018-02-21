import os

from collections import defaultdict as dd
from math import log

from tools.simple_wikipedia.wikipedia_simple import Wikipedia

import config
from utils import Singleton
DATA_PATH = config.swt_data_path

TITLE_MULT = 2.5
TITLE_MULT_FOR_DZ = 2.5

PAGERANK_VAL = 0.3
EXACT_WORDS = 10
ALL_WORDS = 2
REPETITION_PENALTY = 30
FOLLOW_UP_BONUS = 0.1
DEFINITION_BONUS = 0.2
PRONOUN_PENALTY = 0.08
FOLLOW_UP_THR = 0.5
WORD_IN_TITLE_BONUS = 0.5
BAD_TITLE_PENALTY = 2.0
PAGERANK_DZ_VAL = 0.1

def lower(s):
  if not s:
      s = u''
  if not isinstance(s, unicode):
      s = unicode(s, 'utf8')
  return s.lower().encode('utf8')


class TitleFinder(Singleton):
  def __init__(self):
    self.WC = Wikipedia(os.path.join(DATA_PATH, 'simple'))
    self.T  = Wikipedia(os.path.join(DATA_PATH, 'simple_titles'))

    simple_wiki_pagerank = {}
    for x in open(os.path.join(DATA_PATH, 'pagerank.sw.txt')):
      val,t = x.split()
      simple_wiki_pagerank[t] = float(val) ** 0.5

    self.pagerank = simple_wiki_pagerank

    self.repeated = dd(lambda:0)

    self.max_idf = 0
    #for w in self.WC.idf.db: TODO: how to iterate over dbc
    #  self.max_idf = max(self.max_idf, float(self.WC.idf[w]))
    self.max_idf = 12.923797798

  def idf_sum(self, L):
    s = 0.001
    for w in L:
      if w in self.WC.idf:
        s += self.WC.idf[w] # 0 does not count -- maybe bad idea!
    return s

  def get_idf(self, w):
    if type(w) == unicode:
      w = w.encode('utf8')
    if w in self.WC.idf:
      return float(self.WC.idf[w])
    else:
      return float(self.max_idf)


  def idf_term_score(self, doc_zero):
    tf = dd(lambda:0)
    for w in doc_zero:
        tf[w] += 1

    for w in doc_zero:
      tf[w] = (1 + log(tf[w]))

    res = []

    for w in tf:
      if w in self.WC.idf:
        res.append( (self.WC.idf[w] * tf[w], w) )

    res.sort()
    res.reverse()
    return res

  def ban(self, title):
    #print "Banning:", title
    if title is not None:
      self.repeated[lower(title)] += REPETITION_PENALTY

  def query(self, Q, **arg):
    if 'can_repeat' in arg and arg['can_repeat']:
      can_repeat = True
    else:
      can_repeat = False

    ps = self.WC.paragraph_score(Q)
    title_results = dd(lambda:[])
    title_score = dd(lambda:0.0)
    query_words = set(Q)

    for p,s in ps.items():
      t = self.WC.paragraph_title(p)
      title_results[t].append(s)

    for t in title_results:
      ss = sorted(title_results[t])
      ss.reverse()
      mult = 1
      for i in range(len(ss)):
        ss[i] *= mult
        mult *= 0.1

      title_score[t] = sum(ss)

    for p,s in self.T.paragraph_score(Q).items():
      t = self.T.paragraph_title(p)
      title_score[t] += TITLE_MULT * s
      words = set(lower(t).split())

      if words == query_words:
        title_score[t] += EXACT_WORDS
      if query_words <= words:
        title_score[t] += ALL_WORDS

    for t in title_score:
      tl = lower(t)
      if tl in self.pagerank:
        title_score[t] += PAGERANK_DZ_VAL * self.pagerank[tl]

      if lower(t) in self.repeated and not can_repeat:
        title_score[t] -= self.repeated[lower(t)]


    if not title_score:
      return ()
    m = max(title_score.values())

    vs = sorted(title_score.values())[::-1]
    tt = vs[9] if len(vs) >= 10 else 0

    for t in title_score:
      if title_score[t] == m:
        return m,t

    return ()

  def find_document_zero(self, Q, **arg):
    ps = self.WC.paragraph_score(Q)
    title_results = dd(lambda:[])
    title_score = dd(lambda:0.0)
    query_words = set(Q)

    for p,s in ps.items():
      t = self.WC.paragraph_title(p)
      title_results[t].append(s)

    for t in title_results:
      ss = sorted(title_results[t])
      ss.reverse()
      mult = 1
      for i in range(len(ss)):
        ss[i] *= mult
        mult *= 0.6
      title_score[t] = sum(ss)

    for p,s in self.T.paragraph_score(Q).items():
      t = self.T.paragraph_title(p)
      title_score[t] += TITLE_MULT_FOR_DZ * s
      words = set(lower(t).split())

    for t in title_score:
      tl = lower(t)
      if tl in self.pagerank:
        title_score[t] += PAGERANK_DZ_VAL * self.pagerank[tl]

    #print title_score
    return title_score

