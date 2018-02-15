import codecs
import os
import re
import nltk
import numpy as np
from pprint import pprint

import hunspell

from nltk.tokenize import TweetTokenizer
from nltk.tokenize.moses import MosesDetokenizer

import config
from tools.embeddings import word2vec
from utils import U


tweet_tokenizer = TweetTokenizer()
moses_detokenizer = MosesDetokenizer()
nonalpha_pattern = re.compile('([\W_]+|[0-9]+)')

punctuations = {
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|',
    '}', '~'}


class Speller(object):
    large_hspell = None

    def __init__(self, extra_words=(), **kwargs):
        super(Speller, self).__init__(**kwargs)
        if Speller.large_hspell is None:
            Speller.large_hspell = hunspell.HunSpell(
                os.path.join(config.hunspell_path, 'en_US.dic_large.utf8'),
                os.path.join(config.hunspell_path, 'en_US.aff.utf8'))

        self.hspell = hunspell.HunSpell(
            os.path.join(config.hunspell_path, 'en_US.dic_sw.utf8'),
            os.path.join(config.hunspell_path, 'en_US.aff.utf8'))
        self.extra_words = set(extra_words)
        for w in self.extra_words:
            self.hspell.add(w.lower())

    def __getstate__(self):
        return dict(extra_words=self.extra_words)

    def __setstate__(self, state):
        self.__init__(**state)

    def _add_word(self, w):
        if not self.hspell.spell(w):
            self.hspell.add(w)
            self.extra_words.add(w)

    def add_words_from(self, text):
        for w in tokenize(text, lowercase_first_in_sentence=False,
                          correct_spelling=False):
            # Heuritic to remove punctuation
            if len(w) > 2:
                self._add_word(w)

    def spell_token(self, word):
        if word in punctuations:
            return word

        h = self.hspell
        wo = word
        wl = wo.lower()
        wc = wo.capitalize()

        # 1. Try if the smal dict know the words with a few casing options
        for w in [wo, wl, wc]:
            if h.spell(w):
                return w  # return the guessed case

        # 2. See if the large dict knows the word
        for w in [wo, wl, wc]:
            if Speller.large_hspell.spell(w) or w in punctuations:
                return w

        # 3. Generate several suggestion lists, for differently cased variants
        sugs_o = h.suggest(wo)
        sugs_l = h.suggest(wl)
        sugs_c = h.suggest(wc)
        all_sugs = sugs_o + sugs_l + sugs_c
        if not all_sugs:
            return wo

        # Accept a cased extra word
        for s in all_sugs:
            if s.lower() in self.extra_words:
                return s

        # Pick the closest (case insensitive) suggestion
        sel_sug = word
        sel_dist = np.inf
        for sugs in [sugs_o, sugs_l, sugs_c]:
            if not sugs:
                continue
            d = levenshtein(wl, sugs[0].lower())
            if d < sel_dist:
                sel_sug = sugs[0]
                sel_dist = d
        if sel_dist <= len(w)//4 + 1:
            return sel_sug
        else:
            return word

    def spell_tokens(self, ws):
        return [self.spell_token(w) for w in ws]

    def spell_sentence(self, sent):
        tokens = tokenize(sent, correct_spelling=False)
        tokens = [self.spell_token(t) for t in tokens]
        ret = U(detokenize(tokens))
        if config.debug:
            print("Spelling '%s' with ew=%s gives '%s'" %
                  (sent, self.extra_words, ret))
        return ret


def sent_tokenize(utt):
    return [s for s in nltk.sent_tokenize(utt)]


def tokenize(s, method='nltk_tweet', lowercase_first_in_sentence=True,
             correct_spelling=True):
    ret = []
    sents = sent_tokenize(s) if lowercase_first_in_sentence else [s]
    for s in sents:
        if method == 'nltk_tweet':
            tokens = [w for w in tweet_tokenizer.tokenize(s)]
        elif method == 'nltk_word':
            tokens = [w for w in nltk.word_tokenize(s)]
        elif method == 'regexp':
            tokens = nonalpha_pattern.sub(' ', utt).lower().split()
        else:
            raise ValueError

        if correct_spelling:
            tokens = Speller().spell_tokens(tokens)

        if (lowercase_first_in_sentence and tokens and
                not word2vec.has_word(tokens[0])):
            tokens[0] = tokens[0][0].lower() + tokens[0][1:]
        ret += tokens
    return ret


def detokenize(tokens):
    return unicode(moses_detokenizer.detokenize(tokens, return_str=True))


# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:],
            np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
            current_row[1:],
            current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def test():
    sentences = [
        "It's a shame; indeed, 'cause I would like it SOOO...'",
        '"Get up!" she said. I yelled in a loudly-calm fashion.',
        "Which is: Monday, Tuesday. Whaat:) What's up? :-) Arrgh! "
        "Yellow. - this is nice.",
        "Ice-cream. Elon Musk. Gandhi. Lech Walesa.",
    ]
    speller = Speller(extra_words=('Elon',))
    for sent in sentences:
        print sent
        print speller.spell_sentence(sent)
