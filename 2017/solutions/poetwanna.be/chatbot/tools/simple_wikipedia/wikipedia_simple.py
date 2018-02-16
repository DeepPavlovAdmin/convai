from collections import defaultdict as dd
from random import randint
from tools.wikipedia.stop_words import punc, stop_words
from tools.wikipedia.index_reader_cdb import IndexReader, CachedDB
from nltk.tokenize import word_tokenize

import sys
import utils

def lower(s):
  return unicode(s,'utf8').lower().encode('utf8')


class Wikipedia(utils.SingletonWithPath):
  def __init__(self, path):
    self.idf = CachedDB(path + '/wiki.idf.cdb', float)

    self.index = IndexReader(path + '/wiki.index.txt', path + '/wiki.index.positions.cdb')

    p_nr = 0
    self.title4paragraph = []
    self.title_for_nr = {}
    self.title_list = []
    title_nr = {}
    self.pos4par = []
    #self.content = open('/home/prych/text_mining/wiki/fp_wiki.txt', 'r')
    self.content = None # File is not uploaded
    self.cache = {}

    for x in open(path + '/positions.txt'):
      L = x.split()
      pos = int(L[-1])
      title = ' '.join(L[:-2])
      if not title in title_nr:
        n = len(title_nr) + 1
        title_nr[title] = n
        self.title_for_nr[n] = title
        self.title_list.append(title)
      self.pos4par.append(pos)

      self.title4paragraph.append(title_nr[title])

  def paragraph_title(self, n):
    return self.title_for_nr[self.title4paragraph[n]]

  def get_titles(self, paragraphs):
    return set([self.paragraph_title(n) for n in paragraphs ])

  def intersection_query(self, Q):
    if not Q:
      return set()

    res = set(self.index.get(Q[0]))

    print Q

    for term in Q[1:]:
      res.intersection_update(self.index.get(term))

    return res

  def lazy_intersection(self, Q, limit):
    if not Q:
      return set()
    Q = Q[:]
    Q.sort( key=lambda x: -self.idf[x] )

    res = set(self.index.get(Q[0]))

    for term in Q[1:]:
      if len(res) <= limit:
        return res # enough for testing

      res.intersection_update(self.index.get(term))

    return res

  def phrase_filter(self, par, q):
    if len(q) == 1:
      return par

    res = set()
    q_str = ' ' + ' '.join(q) + ' '
    for p in par:
      c = ' ' + lower(self.get_content(p)) + ' '
      if q_str in c:
        res.add(p)
    return res

  def phrase_query(self, q):
    ps = self.lazy_intersection(q, 5)
    ps = self.phrase_filter(ps, q)
    return self.get_titles(ps)

  def get_content(self, par):
    position = self.pos4par[par]
    self.content.seek(position)
    #return self.content.read(200).strip()
    C = self.content.readline().strip()
    if C:
      return C

    return 'PROBLEM:' + self.content.readline().strip()

  def get_terms(self, L):
    L = map(lower, L)
    terms = set()
    for w in L:
      if not w in stop_words and not w in punc and w in self.idf:
        terms.add(w)

    for i in range(len(L)-1):
      b = L[i] + '_' + L[i+1]
      if b in self.idf:
        terms.add(b)

    return terms

  def paragraph_score(self, Q):
    PAR_MAX = 10000
    r = []
    terms = self.get_terms(Q)
    max_score = 0.0
    for t in terms:
      max_score += self.idf[t]

    st = set(terms)
    for t in map(lower,Q):
      if not t in st and not t in punc and not t in self.idf:
        max_score += 10.0

    L = sorted([ (self.idf[t], -len(t), t) for t in terms])
    L.reverse()
    #print 'TERMS=', ' '.join([t for (v,ll,t) in L])
    score = {}

    for idf, ll, t in L:
      pars = self.index.get(t)
      #print 'Paragraphs for',t,'=',len(pars)
      if len(pars) > 10 * PAR_MAX and len(score) > 0:
        break
      for p in pars:
        if p in score:
          score[p] += idf
        elif len(score) <= PAR_MAX:
          score[p] = idf


    for p in score:
      score[p] /= max_score
    return score

  def find_paragraphs(self, Q, K):
    if len(Q) == 0:
      return []

    score = self.paragraph_score(Q)
    pss = score.items()
    pss.sort( key = lambda x: -x[1] )

    res = []

    for par,sc in pss[:K]:
      res.append( (sc, self.get_content(par)) )
    return res


if __name__ == "__main__":
  W = Wikipedia('/var/home/gadula/simple_wiki_data/simple')
  print "Wikipedia loaded, write query"
  cache = {}

  for x in sys.stdin:
    Q = word_tokenize(x.lower())
    print 'QUERY:', ' '.join(Q)
    if x in cache:
      r = cache[x]
    else:
      r = W.phrase_query(Q)

    print 'RESULTS:', len(r)
    for w in r:
      print w
    cache[x] = r

  sys.exit(0)
  for x in sys.stdin:
    Q = x.split()
    for score, content in W.find_paragraphs(Q, 5):
      print score, content

  sys.exit(0)

  for x in sys.stdin:
    Q = x.split()
    #pars = sorted(W.intersection_query(Q))
    pars = sorted(W.lazy_intersection(Q,5))

    for t in W.get_titles(pars):
      print '   ',t

    print "CONTENT"

    if not pars:
      print "Sorry, no answers"
    for p in pars:
      print "TITLES", W.get_titles(set([p]))
      print W.get_content(p)
      print 40 * '-'
      print
