'''
Inspired by
https://glowingpython.blogspot.com/2014/09/text-summarization-with-nltk.html
'''

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest

import config
import numpy as np


class FrequencySummarizer:

    def __init__(self, min_cut=0.2, use_idf=True):
        self._min_cut = min_cut
        self._stopwords = set(stopwords.words('english') + list(punctuation))
        self.use_idf = use_idf

        if self.use_idf:
            self.idf = np.load(config.knn_idf)
            self.max_idf = max(self.idf.values())

    def _compute_frequencies(self, word_sent):
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1

        # frequencies normalization and filtering
        m = float(max(freq.values()))
        for w in freq.keys():
            freq[w] /= m
            if freq[w] <= self._min_cut:
                del freq[w]
        return freq

    def summarize(self, text, n):
        sents = sent_tokenize(text)
        n = min(n, len(sents))
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)

        for i, sent in enumerate(word_sent):
            for w in sent:
                sc = self._freq.get(w, 0)
                if self.use_idf:
                    sc *= self.idf.get(w, self.max_idf) / self.max_idf
                ranking[i] += sc

        inds_scores = self._rank(ranking, n)
        max_sc = inds_scores[0][1]
        return [(sc / max_sc, sents[i]) for (i, sc) in inds_scores]

    def _rank(self, ranking, n):
        return nlargest(n, ranking.items(), key=lambda x: x[1])
