from __future__ import print_function

import collections

import config

from .stop_words import punc, stop_words
from .index_reader_cdb import IndexReader, CachedDB
import utils

from time import time
import numpy as np


def lower(s):
    if not isinstance(s, unicode):
        s = unicode(s, 'utf8')
    return s.lower().encode('utf8')


class WikiPositions(utils.SingletonWithPath):

    def __init__(self, path):
        self.title4paragraph = []
        self.title_nr = {}
        self.title_list = []
        self.pos4par = []

        print('Loading Wiki positions from ' + path + '...')
        for x in open(path):
            L = x.split()
            pos = int(L[-1])
            title = ' '.join(L[:-2])
            if title not in self.title_nr:
                self.title_nr[title] = len(self.title_nr)
                self.title_list.append(title)
            self.pos4par.append(pos)
            self.title4paragraph.append(self.title_nr[title])
        print('Wiki positions loaded.')


class Wikipedia(utils.SingletonWithPath):

    def __init__(self, name):
        cfg = config.wiki_paths[name]
        index_dir = cfg['index_dir']
        positions_file = cfg['positions_file']
        tokens_file = cfg['tokens_file']

        self.idf = CachedDB(index_dir + '/wiki.idf.cdb', float)

        self.index = IndexReader(
            index_dir + '/wiki.index.txt',
            index_dir + '/wiki.index.positions.cdb')

        positions = WikiPositions(positions_file)
        self.title4paragraph = positions.title4paragraph
        self.title_nr = positions.title_nr
        self.title_list = positions.title_list
        self.pos4par = positions.pos4par

        if tokens_file is not None:
            self.content = open(tokens_file, 'r')
        else:
            self.content = None

        self.pars4title = collections.defaultdict(lambda: [])
        for par_nr, tit_nr in enumerate(self.title4paragraph):
            self.pars4title[tit_nr].append(par_nr)

    def get_contents4title(self, title, K=100):
        if title not in self.title_nr:
            return []
        if self.title_nr[title] not in self.pars4title:
            return []
        res = []
        for par_nr in self.pars4title[self.title_nr[title]][:K]:
            res.append(self.get_content(par_nr))
        return res

    def paragraph_title(self, n):
        return self.title_list[self.title4paragraph[n]]

    def get_titles(self, paragraphs):
        return set([self.paragraph_title(n) for n in paragraphs])

    def intersection_query(self, Q):
        if not Q:
            return set()

        res = set(self.index.get(Q[0]))
        for term in Q[1:]:
            res.intersection_update(self.index.get(term))

        return res

    def get_content(self, par):
        position = self.pos4par[par]
        self.content.seek(position)
        return self.content.readline().strip()

    def get_content4pos_list(self, pos_list):
        res = []
        for pos in pos_list:
            self.content.seek(pos)
            res.append(self.content.readline().strip())
        return res

    def get_terms(self, L):
        L = map(lower, L)
        terms = set()
        for w in L:
            if w not in stop_words and w not in punc and w in self.idf:
                terms.add(w)

        for i in range(len(L)-1):
            b = L[i] + '_' + L[i+1]
            if b in self.idf:
                terms.add(b)

        return terms

    def find_paragraphs(self, Q, K, main_title=None, main_title_bonus=1.,
                        debug=False):
        t0 = time()

        PAR_MAX = 10000
        if len(Q) == 0:
            return []

        terms = self.get_terms(Q)

        if debug:
            print("        find_pars get_terms:", time()-t0)
            t0 = time()

        max_score = 0.0
        for t in terms:
            max_score += self.idf[t]

        L = sorted([(self.idf[t], -len(t), t) for t in terms])
        L.reverse()

        if debug:
            print('        TERMS=', ' '.join([t for (_, _, t) in L]))
        score = {}

        if debug:
            print("        find_pars before loop:", time()-t0)
            t0 = time()

        for idf, _, t in L:
            pars = self.index.get(t)
            if debug:
                print('        Paragraphs for', t, '=', len(pars))
            if len(pars) > 10 * PAR_MAX and len(score) > 0:
                break
            for p in pars:
                if p in score:
                    score[p] += idf
                elif len(score) <= PAR_MAX:
                    score[p] = idf

        if debug:
            print("        len(score) =", len(score))
            print("        find_pars loop:", time()-t0)
            t0 = time()

        if main_title is not None and debug:
            print('Bonus %f for the title: %s' %
                  (main_title_bonus, main_title))

        main_title_nr = self.title_nr.get(main_title, None)
        if main_title_nr is not None:
            for p in score:
                if self.title4paragraph[p] == main_title_nr:
                    score[p] *= main_title_bonus

        if debug:
            print("        find_pars score mods:", time()-t0)
            t0 = time()

        keys = score.keys()
        values = score.values()

        if debug:
            print("        find_pars '.items()':", time()-t0)
            t0 = time()

        K = min(K, len(score))
        topK = np.argpartition(values, -K)[-K:]

        pss = [(keys[i], values[i]) for i in topK]
        pss_pos = sorted([(self.pos4par[par], par, sc) for (par, sc) in pss])

        if debug:
            print("        find_pars sorting:", time()-t0)
            t0 = time()

        contents = self.get_content4pos_list([x[0] for x in pss_pos])
        titles = [self.paragraph_title(x[1]) for x in pss_pos]

        res = []

        for i in range(len(pss_pos)):
            res.append((pss_pos[i][2] / max_score, contents[i], titles[i]))

        if debug:
            print("        find_pars res prep:", time()-t0)
            t0 = time()

        return sorted(res, reverse=True)

    def find_titles(self, Q, K, debug=False):
        t0 = time()

        PAR_MAX = 1000
        if len(Q) == 0:
            return []

        terms = self.get_terms(Q)

        if debug:
            print("        find_titles get_terms:", time()-t0)
            t0 = time()

        max_score = 0.0
        for t in terms:
            max_score += self.idf[t]

        L = sorted([(self.idf[t], -len(t), t) for t in terms])
        L.reverse()

        score = {}

        if debug:
            print("        find_titles before loop:", time()-t0)
            t0 = time()

        for idf, _, t in L:
            pars = self.index.get(t)

            if len(pars) > 10 * PAR_MAX and len(score) > 0:
                break
            for p in pars:
                if p in score:
                    score[p] += idf
                elif len(score) <= PAR_MAX:
                    score[p] = idf

        if debug:
            print("        len(score) =", len(score))
            print("        find_titles loop:", time()-t0)
            t0 = time()

        keys = score.keys()
        values = score.values()

        if debug:
            print("        find_titles '.items()':", time()-t0)
            t0 = time()

        K = min(K, len(score))
        topK = np.argpartition(values, -K)[-K:]

        pss = sorted([(keys[i], values[i]) for i in topK],
                     key=lambda x: -x[1])

        if debug:
            print("        find_titles sorting:", time()-t0)
            t0 = time()

        res = []

        for par, sc in pss:
            res.append((sc / max_score, self.paragraph_title(par)))
        if debug:
            print("        find_titles res prep:", time()-t0)
            t0 = time()
        return res

    def get_main_title(self, Q, debug=False):
        titles = self.find_titles(Q, 1, debug=debug)
        if not titles:
            return None, None
        scr, title = titles[0]
        return scr, title

    def get_main_article(self, Q, K, debug=False):
        scr, main_title = self.get_main_title(Q, debug=debug)
        if main_title is None:
            return None, None, None
        return scr, self.get_contents4title(main_title, K), main_title
