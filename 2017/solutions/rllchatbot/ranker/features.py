from nltk.corpus import stopwords
from nltk import ngrams, word_tokenize, sent_tokenize
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from embedding_metrics import w2v
from embedding_metrics import greedy_score, extrema_score, average_score

import time
import os
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

import spacy
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

"""
This file defines different hand-enginered features based
on the conversation article, the conversation hisory, and
one candidate response.
These features will then be used as input signals to a
fully-connected feed-foward neural network to predict either
the full dialogue score or the utterance score.

continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit
"""

logger.info("loading nltk english stop words...")
stop = set(stopwords.words('english'))
logger.info(stop)
logger.info("")


def initialize_features(feature_list):
    """
    construct a list of Feature instances to be used for all responses
    :param feature_list: list of feature names (str)
    :return: list of `Feature` instances, and the total dimension
    """
    feature_objects = []  # list of feature objects
    dim = 0

    for f in feature_list:
        feature = eval(f)(article=None, context=None, candidate=None)
        dim += feature.dim
        feature_objects.append(feature)

    if len(feature_objects) == 0:
        print "WARNING: no feature recognized in %s" % (feature_list,)

    return feature_objects, dim


def get(feature_objects, dim, article, context, candidate):
    """
    get all features we want for a triple of (article, context, candidate response).
    :param feature_objects: list of `Feature` instances to measure for the above triple.
    :param dim: total dimension of all features
    :param article: the text of the conversation article to talk to
    :param context: the list of user & bot utterances so far
    :param candidate: the candidate response one model proposed
    :return: an aray containing all feature objects you requested for.
    """
    raw_features = np.zeros((dim,))
    idx = 0
    for f in feature_objects:
        f.set(article, context, candidate)  # compute feature
        if f.feat is None or len(f.feat) != f.dim:
            logger.warning("unable to compute feature %s" % f.__class__.__name__)
            logger.warning("dim: %d --- feat: %s" % (f.dim, f.feat))
        else:
            raw_features[idx: idx+f.dim] = f.feat  # set raw features
        idx += f.dim

    return raw_features


#####################
### GENERIC CLASS ###
#####################

class Feature(object):

    def __init__(self, dim, article=None, context=None, candidate=None):
        self.dim = dim
        self.feat = None
        
    def set(self, article, context, candidate):
        """
        To be implemented in each sub-class
        """
        pass


############################
### SPECIFIC SUB-CLASSES ###
############################

### Average embedding ###

class AverageWordEmbedding_Candidate(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(AverageWordEmbedding_Candidate, self).__init__(w2v.vector_size, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the candidate response
        """
        if candidate is None:
            self.feat = None
        else:
            candidate = candidate.lower()

            X = np.zeros((self.dim,), dtype='float32')
            for tok in word_tokenize(candidate):
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
                X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class AverageWordEmbedding_User(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(AverageWordEmbedding_User, self).__init__(w2v.vector_size, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the last user turn
        """
        if context is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            for tok in word_tokenize(context[-1].lower()):
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
                X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class AverageWordEmbedding_LastK(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(AverageWordEmbedding_LastK, self).__init__(w2v.vector_size, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the last k turns
        """
        if context is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            content = ' '.join(context[-self.k:]).lower()
            logger.debug("last %d turns: %s" % (self.k, content))
            for tok in word_tokenize(content):
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
                X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class AverageWordEmbedding_kUser(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(AverageWordEmbedding_kUser, self).__init__(w2v.vector_size, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the last k user turns
        """
        if context is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            start = min(len(context), 2*self.k+1)
            content = np.array(context)[range(-start, 0, 2)]
            content = ' '.join(content).lower()
            logger.debug("last %d user turns: %s" % (self.k, content))
            for tok in word_tokenize(content):
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
                X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class AverageWordEmbedding_Article(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(AverageWordEmbedding_Article, self).__init__(w2v.vector_size, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the article
        """
        if article is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            for tok in word_tokenize(article.lower()):
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
                X = np.array(X)/np.linalg.norm(X)
            self.feat = X


### Candidate -- user turn match ###

class Similarity_CandidateUser(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=3
        super(Similarity_CandidateUser, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to:
        - greedy score (dim: 1) between candidate response & last user turn
        - average embedding score (dim: 1) between candidate response & last user turn
        - extrema embedding score (dim: 1) between candidate response & last user turn
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_turn = context[-1].lower()

            res1 = greedy_score(candidate, last_turn)
            res2 = greedy_score(last_turn, candidate)
            self.feat = np.zeros(3)
            self.feat[0] = (res1 + res2) / 2.0
            self.feat[1] = float(average_score(candidate, last_turn))
            self.feat[2] = float(extrema_score(candidate, last_turn))


### Candidate -- last k turns match ###

class Similarity_CandidateLastK(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=3
        super(Similarity_CandidateLastK, self).__init__(3, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to:
        - greedy score (dim: 1) between candidate response & last k turns
        - average embedding score (dim: 1) between candidate response & last k turns
        - extrema embedding score (dim: 1) between candidate response & last k turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_turns = ' '.join(context[-self.k:]).lower()
            logger.debug("last %d turns: %s" % (self.k, last_turns))

            res1 = greedy_score(candidate, last_turns)
            res2 = greedy_score(last_turns, candidate)
            self.feat = np.zeros(3)
            self.feat[0] = (res1 + res2) / 2.0
            self.feat[1] = float(average_score(candidate, last_turns))
            self.feat[2] = float(extrema_score(candidate, last_turns))


### Candidate -- last k turns without stop words match ###

class Similarity_CandidateLastK_noStop(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=3
        super(Similarity_CandidateLastK_noStop, self).__init__(3, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to:
        - greedy score (dim: 1) between candidate response & last k turns without stop words
        - average embedding score (dim: 1) between candidate response & last k turns without stop words
        - extrema embedding score (dim: 1) between candidate response & last k turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_turns = ' '.join(context[-self.k:]).lower()
            last_turns = ' '.join(filter(lambda word: word not in stop, word_tokenize(last_turns)))
            logger.debug("last %d turns: %s" % (self.k, last_turns))

            res1 = greedy_score(candidate, last_turns)
            res2 = greedy_score(last_turns, candidate)
            self.feat = np.zeros(3)
            self.feat[0] = (res1 + res2) / 2.0
            self.feat[1] = float(average_score(candidate, last_turns))
            self.feat[2] = float(extrema_score(candidate, last_turns))


### Candidate -- last k user turns match ###

class Similarity_CandidateKUser(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=3
        super(Similarity_CandidateKUser, self).__init__(3, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to:
        - greedy score (dim: 1) between candidate response & last k user turns
        - average embedding score (dim: 1) between candidate response & last k user turns
        - extrema embedding score (dim: 1) between candidate response & last k user turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            start = min(len(context), 2*self.k+1)
            user_turns = np.array(context)[range(-start, 0, 2)]
            user_turns = ' '.join(user_turns).lower()
            logger.debug("last %d user turns: %s" % (self.k, user_turns))

            res1 = greedy_score(candidate, user_turns)
            res2 = greedy_score(user_turns, candidate)
            self.feat = np.zeros(3)
            self.feat[0] = (res1 + res2) / 2.0
            self.feat[1] = float(average_score(candidate, user_turns))
            self.feat[2] = float(extrema_score(candidate, user_turns))


### Candidate -- last k user turns without stop words match ###

class Similarity_CandidateKUser_noStop(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=3
        super(Similarity_CandidateKUser_noStop, self).__init__(3, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to:
        - greedy score (dim: 1) between candidate response & last k user turns without stop words
        - average embedding score (dim: 1) between candidate response & last k user turns without stop words
        - extrema embedding score (dim: 1) between candidate response & last k user turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            start = min(len(context), 2*self.k+1)
            user_turns = np.array(context)[range(-start, 0, 2)]
            user_turns = ' '.join(user_turns).lower()
            user_turns = ' '.join(filter(lambda word: word not in stop, word_tokenize(user_turns)))
            logger.debug("last %d user turns: %s" % (self.k, user_turns))

            res1 = greedy_score(candidate, user_turns)
            res2 = greedy_score(user_turns, candidate)
            self.feat = np.zeros(3)
            self.feat[0] = (res1 + res2) / 2.0
            self.feat[1] = float(average_score(candidate, user_turns))
            self.feat[2] = float(extrema_score(candidate, user_turns))


### Candidate -- article match ###

class Similarity_CandidateArticle(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=3
        super(Similarity_CandidateArticle, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to:
        - greedy score (dim: 1) between candidate response & article
        - average embedding score (dim: 1) between candidate response & article
        - extrema embedding score (dim: 1) between candidate response & article
        """
        if candidate is None or article is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            article = article.lower()

            res1 = greedy_score(candidate, article)
            res2 = greedy_score(article, candidate)
            self.feat = np.zeros(3)
            self.feat[0] = (res1 + res2) / 2.0
            self.feat[1] = float(average_score(candidate, article))
            self.feat[2] = float(extrema_score(candidate, article))


### Candidate -- article without stop words match ###

class Similarity_CandidateArticle_noStop(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=3
        super(Similarity_CandidateArticle_noStop, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to:
        - greedy score (dim: 1) between candidate response & article without stop words
        - average embedding score (dim: 1) between candidate response & article without stop words
        - extrema embedding score (dim: 1) between candidate response & article without stop words
        """
        if candidate is None or article is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            article = article.lower()
            article = ' '.join(filter(lambda word: word not in stop, word_tokenize(article)))

            res1 = greedy_score(candidate, article)
            res2 = greedy_score(article, candidate)
            self.feat = np.zeros(3)
            self.feat[0] = (res1 + res2) / 2.0
            self.feat[1] = float(average_score(candidate, article))
            self.feat[2] = float(extrema_score(candidate, article))


### n-gram & entity overlaps ###

class NonStopWordOverlap(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(NonStopWordOverlap, self).__init__(1, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        0 / 1 indicating if response has at least one word overlapping with:
        - previous user turn (binary feature size: 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_response = context[-1].lower()

            candidate_tokens = set(filter(lambda word: word not in stop, word_tokenize(candidate)))
            last_response_tokens = set(filter(lambda word: word not in stop, word_tokenize(last_response)))

            self.feat = np.array([0])
            if len(candidate_tokens.intersection(last_response_tokens)) > 0:
                self.feat = np.array([1])


class BigramOverlap(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(BigramOverlap, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        0 / 1 indicating if response has at least one bigram overlapping with:
        - the previous user turn (binary feature size: 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        - any previous turn (binary feature size: 1) -- for f_pi(a, h, i)
        - the article (binary feature size: 1)
        """
        if candidate is None or context is None or article is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            article = article.lower()
            last_response = context[-1].lower()
            content = ' '.join(context).lower()

            candidate_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(candidate), 2)])
            last_response_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(last_response), 2)])
            content_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(content), 2)])
            article_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(article), 2)])

            self.feat = np.zeros(3)
            if len(candidate_bigrams.intersection(last_response_bigrams)) > 0:
                self.feat[0] = 1
            if len(candidate_bigrams.intersection(content_bigrams)) > 0:
                self.feat[1] = 1
            if len(candidate_bigrams.intersection(article_bigrams)) > 0:
                self.feat[2] = 1


class TrigramOverlap(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(TrigramOverlap, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        0 / 1 indicating if response has at least one trigram overlapping with:
        - the previous user turn (binary feature size: 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        - any previous turn (binary feature size: 1) -- for f_pi(a, h, i)
        - the article (binary feature size: 1)
        """
        if candidate is None or context is None or article is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            article = article.lower()
            last_response = context[-1].lower()
            content = ' '.join(context).lower()

            candidate_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(candidate),3)])
            last_response_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(last_response), 3)])
            content_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(content), 3)])
            article_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(article), 3)])

            self.feat = np.zeros(3)
            if len(candidate_bigrams.intersection(last_response_bigrams)) > 0:
                self.feat[0] = 1
            if len(candidate_bigrams.intersection(content_bigrams)) > 0:
                self.feat[1] = 1
            if len(candidate_bigrams.intersection(article_bigrams)) > 0:
                self.feat[2] = 1


class EntityOverlap(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(EntityOverlap, self).__init__(3, article, context, candidate)
        self.nlp = spacy.load('en')
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        0 / 1 indicating if response has at least one entity overlapping with:
        - the previous user turn (binary feature size: 1) -- for f_pi(a, h, i)
        - any previous turn (binary feature size: 1) -- for f_pi(a, h, i)
        - the article (binary feature size: 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        """
        if candidate is None or context is None or article is None:
            self.feat = None
        else:
            content = ' '.join(context).lower()
            article = article.lower()
            candidate = candidate.lower()
            last_response = context[-1].lower()

            content = self.nlp(unicode(content))
            article = self.nlp(unicode(article))
            candidate = self.nlp(unicode(candidate))
            last_response = self.nlp(unicode(last_response))

            content_entities = set([ent.label_ for ent in content.ents])
            article_entities = set([ent.label_ for ent in article.ents])
            candidate_entities = set([ent.label_ for ent in candidate.ents])
            last_response_entities = set([ent.label_ for ent in last_response.ents])

            self.feat = np.zeros(3)
            if len(candidate_entities.intersection(last_response_entities)) > 0:
                self.feat[0] = 1
            if len(candidate_entities.intersection(content_entities)) > 0:
                self.feat[1] = 1
            if len(candidate_entities.intersection(article_entities)) > 0:
                self.feat[2] = 1


### generic turns ###

class GenericTurns(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(GenericTurns, self).__init__(2, article, context, candidate)
        self.generic_list = []
        with open('/root/convai/data/generic_list.txt') as fp:
        # with open('%s/../data/generic_list.txt' % path, 'r') as fp:
            for line in fp:
                self.generic_list.append(line.strip())
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        0 / 1 indicating if:
        - the candidate has only stop words or words of 3 characters or less
        - the last turn has only stop words or words of 3 characters or less
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_response = context[-1].lower()

            candidate_tokens = word_tokenize(candidate)
            last_resp_tokens = word_tokenize(last_response)

            generic_candidate = len([word for word in candidate_tokens if word in self.generic_list])
            generic_last_resp = len([word for word in last_resp_tokens if word in self.generic_list])
            self.feat = np.zeros(2)
            if generic_candidate == len(candidate_tokens):
                self.feat[0] = 1
            if generic_last_resp == len(last_resp_tokens):
                self.feat[1] = 1


### word presence ###

class WhWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(WhWords, self).__init__(2, article, context, candidate)
        self.set(article, context, candidate)

    def is_wh(self, word):
        return word in ['who', 'whos', 'where', 'wheres', 'when', 'whens',
                        'why', 'whys', 'what', 'whats', 'which', 'how', 'hows']

    def set(self, article, context, candidate):
        """
        0 / 1 indicating if the turn has a word starting with "wh", on the:
        - candidate response (scalar of size 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        - last user turn (scalar of size 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_response = context[-1].lower()

            candidate_tokens = word_tokenize(candidate)
            last_resp_tokens = word_tokenize(last_response)

            wh_candidate = len([word for word in candidate_tokens if self.is_wh(word)])
            wh_last_resp = len([word for word in last_resp_tokens if self.is_wh(word)])
            self.feat = np.zeros(2)
            if wh_candidate > 0:
                self.feat[0] = 1
            if wh_last_resp > 0:
                self.feat[1] = 1


class IntensifierWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(IntensifierWords, self).__init__(4, article, context, candidate)
        self.intensifier_list = []
        with open('/root/convai/data/intensifier_list.txt') as fp:
        # with open('%s/../data/intensifier_list.txt' % path, 'r') as fp:
            for line in fp:
                self.intensifier_list.append(line.strip())
        self.set(article, context, candidate)

    def count(self, text):
        counter = 0
        for phrase in self.intensifier_list:
            if phrase in text:
                counter += 1
        return counter

    def set(self, article, context, candidate):
        """
         4 features:
         1: candidate - boolean value if the candidate response contains any words from the list
         2: candidate - float percentage of words which are flagged in the intensifier list
         3: last response - boolean value if the candidate response contains any words from the list
         4: last response - float percentage of words which are flagged in the intensifier list
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_response = context[-1].lower()

            candidate_intensifier_words_count = self.count(candidate)
            last_response_intensifier_words_count = self.count(last_response)

            self.feat = np.zeros(4)
            if candidate_intensifier_words_count > 0:
                self.feat[0] = 1
            if last_response_intensifier_words_count > 0:
                self.feat[2] = 1
            self.feat[1] = (1.0 * candidate_intensifier_words_count) / len(word_tokenize(candidate))
            self.feat[3] = (1.0 * last_response_intensifier_words_count) / len(word_tokenize(last_response))


class ConfusionWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(ConfusionWords, self).__init__(4, article, context, candidate)
        self.confusion_list = []
        with open('/root/convai/data/confusion_list.txt') as fp:
        # with open('%s/../data/confusion_list.txt' % path, 'r') as fp:
            for line in fp:
                self.confusion_list.append(line.strip())
        self.set(article, context, candidate)

    def count(self, text):
        counter = 0
        for phrase in self.confusion_list:
            if phrase in text:
                counter += 1
        return counter

    def set(self, article, context, candidate):
        """
         4 features:
         1: candidate - boolean value if the candidate response contains any words from the list
         2: candidate - float percentage of words which are flagged in the confusion list
         3: last response - boolean value if the candidate response contains any words from the list
         4: last response - float percentage of words which are flagged in the confusion list
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_response = context[-1].lower()

            candidate_confusion_words_count = self.count(candidate)
            last_response_confusion_words_count = self.count(last_response)

            self.feat = np.zeros(4)
            if candidate_confusion_words_count > 0:
                self.feat[0] = 1
            if last_response_confusion_words_count > 0:
                self.feat[2] = 1
            self.feat[1] = (1.0 * candidate_confusion_words_count) / len(word_tokenize(candidate))
            self.feat[3] = (1.0 * last_response_confusion_words_count) / len(word_tokenize(last_response))


class ProfanityWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(ProfanityWords, self).__init__(4, article, context, candidate)
        self.profanity_list = []
        with open('/root/convai/data/profanity_list.txt') as fp:
        # with open('%s/../data/profanity_list.txt' % path, 'r') as fp:
            for line in fp:
                self.profanity_list.append(line.strip())
        self.set(article, context, candidate)

    def count(self, text):
        counter = 0
        for phrase in self.profanity_list:
            if phrase in text:
                counter += 1
        return counter

    def set(self, article, context, candidate):
        """
         4 features:
         1: candidate - boolean value if the candidate response contains any words from the list
         2: candidate - float percentage of words which are flagged in the profanity list
         3: last response - boolean value if the candidate response contains any words from the list
         4: last response - float percentage of words which are flagged in the profanity list
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_response = context[-1].lower()

            candidate_hate_words_count = self.count(candidate)
            last_response_hate_words_count = self.count(last_response)

            self.feat = np.zeros(4)
            if candidate_hate_words_count > 0:
                self.feat[0] = 1
            if last_response_hate_words_count > 0:
                self.feat[2] = 1
            self.feat[1] = (1.0 * candidate_hate_words_count) / len(word_tokenize(candidate))
            self.feat[3] = (1.0 * last_response_hate_words_count) / len(word_tokenize(last_response))


class Negation(Feature):
    def __init__(self, article, context, candidate):
        super(Negation, self).__init__(2, article, context, candidate)
        self.set(article, context, candidate)

    def is_negation(self, word):
        return word == "not" or word.endswith("n't")

    def set(self, article, context, candidate):
        """
        0 / 1 indicating if the turn has a negation word ('not' or "n't") on the:
        - candidate response (scalar of size 1)
        - user response (scalar of size 1)
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            last_turn = context[-1].lower()

            candidate_tokens = [self.is_negation(word) for word in word_tokenize(candidate)]
            last_turn_tokens = [self.is_negation(word) for word in word_tokenize(last_turn)]

            self.feat = np.zeros(2)
            if np.any(candidate_tokens):
                self.feat[0] = 1
            if np.any(last_turn_tokens):
                self.feat[1] = 1


### length ###

class DialogLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(DialogLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        number of turns so far n, sqrt(n), log(n) (3 scalars: size 3) -- for g_phi(a, h, i)
        """
        if context is None:
            self.feat = None
        else:
            self.feat = np.zeros(3)
            self.feat[0] = len(context)
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of turns in context is zero: `%s`" % context
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


class LastUserLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(LastUserLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         number of words n, sqrt(n), log(n) (3 scalars: size 3) -- for g_phi(a, h, i)
        """
        if context is None:
            self.feat = None
        else:
            last_user_turn = context[-1]
            self.feat = np.zeros(3)
            self.feat[0] = len(word_tokenize(last_user_turn))
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of words in last user msg is zero: `%s`" % last_user_turn
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


class CandidateLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(CandidateLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        number of words n, sqrt(n), log(n) (3 scalars: dim 3)
        """
        if candidate is None:
            self.feat = None
        else:
            self.feat = np.zeros(3)
            self.feat[0] = len(word_tokenize(candidate))
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of words in candidate is zero: `%s`" % candidate
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


class ArticleLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(ArticleLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         number of sentences in the article n, sqrt(n), log(n) (3 scalars: size 3) -- for g_phi(a, h, i)
        """
        if article is None:
            self.feat = None
        else:
            article_sents = sent_tokenize(article)
            self.feat = np.zeros(3)
            self.feat[0] = len(article_sents)
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of sentences in article is zero: `%s`" % article
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


### Dialog ACT ###

class DialogActCandidate(Feature):
    greeting_words = ('hello', 'hi', 'hey', 'bye', 'goodbye')
    personal_q_bigrams = (
        'do you', 'did you', 'will you', 'would you',
        'can you', 'could you', 'you like'
    )
    affirmative_words = (
        'yes', 'yea', 'yep', 'yop', 'yeah', 'right',
        'mm', 'mmm', 'mmmm', 'mmmm')
    negative_words = ('no', 'noo', 'nooo', 'noooo', 'nop', 'nope')
    request_unigrams = ('please',)
    request_bigrams = (
        'can you', 'could you', "let 's", "talk about", "make me", "chat about",
        "discuss about", "tell me", "be quiet", "shut up", "have you"
    )
    request_trigrams = (
        "ca n't you", "could n't you", "have n't you", "talk to me",
        "chat with me", "converse with me"
    )
    political_words = (
        'trump', 'donald', 'hillary', 'clinton', 'obama', 'barack', 'president',
        'minister', 'senate', 'senator', 'bush', 'china', 'russia', 'iran', 'politics',
        'war', 'weapon', 'weapons'
    )

    def __init__(self, article=None, context=None, candidate=None):
        super(DialogActCandidate, self).__init__(6, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        1 hot encoding vector for the candidate act. we consider:
        GREETING / PERSONAL_Q / AFFIRMATIVE / NEGATIVE / REQUEST / POLITIC
        """
        if candidate is None:
            self.feat = None
        else:
            candidate = candidate.lower()
            candidate_tokens = word_tokenize(candidate)
            candidate_bigrams = set([' '.join(grams) for grams in ngrams(candidate_tokens, 2)])
            candidate_trigrams = set([' '.join(grams) for grams in ngrams(candidate_tokens, 3)])

            greeting = 1 if len(set(candidate_tokens).intersection(self.greeting_words)) > 0 else 0
            personal_q = 1 if len(candidate_bigrams.intersection(self.personal_q_bigrams)) > 0 else 0
            affirmative = 1 if len(set(candidate_tokens).intersection(self.affirmative_words)) > 0 else 0
            negative = 1 if len(set(candidate_tokens).intersection(self.negative_words)) > 0 else 0
            request = 1 if (
                len(set(candidate_tokens).intersection(self.request_unigrams)) > 0
                or len(candidate_bigrams.intersection(self.request_bigrams)) > 0
                or len(candidate_trigrams.intersection(self.request_trigrams)) > 0
            ) else 0
            politic = 1 if len(set(candidate_tokens).intersection(self.political_words)) > 0 else 0

            self.feat = np.array([greeting, personal_q, affirmative, negative, request, politic])


class DialogActLastUser(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(DialogActLastUser, self).__init__(6, article, context, candidate)
        # NOTE: use the same unigram/bigram/trigrams as the previous class
        self.greeting_words = DialogActCandidate.greeting_words
        self.personal_q_bigrams = DialogActCandidate.personal_q_bigrams
        self.affirmative_words = DialogActCandidate.affirmative_words
        self.negative_words = DialogActCandidate.negative_words
        self.request_unigrams = DialogActCandidate.request_unigrams
        self.request_bigrams = DialogActCandidate.request_bigrams
        self.request_trigrams = DialogActCandidate.request_trigrams
        self.political_words = DialogActCandidate.political_words

        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        1 hot encoding vector for the last turn act. we consider:
        GREETING / PERSONAL_Q / AFFIRMATIVE / NEGATIVE / REQUEST / POLITIC
        """
        if context is None:
            self.feat = None
        else:
            last_turn = context[-1].lower()
            last_turn_tokens = word_tokenize(last_turn)
            last_turn_bigrams = set([' '.join(grams) for grams in ngrams(last_turn_tokens, 2)])
            last_turn_trigrams = set([' '.join(grams) for grams in ngrams(last_turn_tokens, 3)])

            greeting = 1 if len(set(last_turn_tokens).intersection(self.greeting_words)) > 0 else 0
            personal_q = 1 if len(last_turn_bigrams.intersection(self.personal_q_bigrams)) > 0 else 0
            affirmative = 1 if len(set(last_turn_tokens).intersection(self.affirmative_words)) > 0 else 0
            negative = 1 if len(set(last_turn_tokens).intersection(self.negative_words)) > 0 else 0
            request = 1 if (
                len(set(last_turn_tokens).intersection(self.request_unigrams)) > 0
                or len(last_turn_bigrams.intersection(self.request_bigrams)) > 0
                or len(last_turn_trigrams.intersection(self.request_trigrams)) > 0
            ) else 0
            politic = 1 if len(set(last_turn_tokens).intersection(self.political_words)) > 0 else 0

            self.feat = np.array([greeting, personal_q, affirmative, negative, request, politic])


### Sentiment score ###

class SentimentScoreCandidate(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(SentimentScoreCandidate, self).__init__(3, article, context, candidate)
        self.analyzer = SentimentIntensityAnalyzer()
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         3 features: one hot vector for candidate to be positive, negative or neutral
        """
        if candidate is None:
            self.feat = None
        else:
            candidate_vs = self.analyzer.polarity_scores(candidate)
            self.feat = np.zeros(3)
            if candidate_vs['compound'] >= 0.5:  # positive
                self.feat[0] = 1
            elif candidate_vs['compound'] <= -0.5:  # negative
                self.feat[1] = 1
            else:
                self.feat[2] = 1  # neutral


class SentimentScoreLastUser(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(SentimentScoreLastUser, self).__init__(3, article, context, candidate)
        self.analyzer = SentimentIntensityAnalyzer()
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         3 features: one hot vector for last user turn to be positive, negative or neutral
        """

        if context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            last_response_vs = self.analyzer.polarity_scores(last_response)
            self.feat = np.zeros(3)
            if last_response_vs['compound'] >= 0.5: # positive
                self.feat[0] = 1
            elif last_response_vs['compound'] <= -0.5: # negative
                self.feat[1] = 1
            else:
                self.feat[2] = 1 # neutral


if __name__ == '__main__':

    article = "russia asks facebook to comply with personal data policy friday, september 29, 2017 \
        on tuesday, russian government internet watchdog roskomnadzor 'insisted' us - based social \
        networking website facebook comply with law # 242 on personal data of users in order to \
        continue operating in the country . per law # 242 , user data of russian citizens should be \
        hosted on local servers - the rule which business - oriented networking site linkedin did \
        not agree to, for which linkedin was eventually blocked in the country."

    context = ["hello user ! this article is very interesting don 't you think ?",
        "hello chat bot ! yes indeed . looks like russia starts to apply the same rules as china",
        "yeah i don 't know about that .",
        "facebook should be available everywhere in the world",
        "yeah i don 't know about that .",
        "you don 't know much do you ? ",
        "i am not a fan of russian policies",
        "haha me neither !"
    ]

    features = [
        'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_LastK',
        'AverageWordEmbedding_kUser', 'AverageWordEmbedding_Article',
        'Similarity_CandidateUser',
        'Similarity_CandidateLastK', 'Similarity_CandidateLastK_noStop',
        'Similarity_CandidateKUser', 'Similarity_CandidateKUser_noStop',
        'Similarity_CandidateArticle', 'Similarity_CandidateArticle_noStop',
        'NonStopWordOverlap', 'BigramOverlap', 'TrigramOverlap', 'EntityOverlap',
        'GenericTurns',
        'WhWords', 'IntensifierWords', 'ConfusionWords', 'ProfanityWords', 'Negation',
        'DialogLength', 'LastUserLength', 'CandidateLength', 'ArticleLength',
        'DialogActCandidate', 'DialogActLastUser',
        'SentimentScoreCandidate', 'SentimentScoreLastUser'
    ]
    # candidate1 = "i am happy to make you laugh"
    # candidate2 = "ha ha ha"
    # candidate3 = "i am not a fan of you"
    # for feature in features:
    #     logger.info("class: %s" % feature)
    #     feature_obj = get(article, context, candidate3, [feature])[0]
    #     logger.info("feature: %s" % (feature_obj.feat,))
    #     logger.info("dim: %d" % feature_obj.dim)
    #     logger.info("")

    logger.info("creating features...")
    start_creation_time = time.time()
    feat_objects, dim = initialize_features(features)
    logger.info("created all feature instances in %s sec" % (time.time() - start_creation_time))
    logger.info("total dim: %d" % dim)

    while True:
        candidate = raw_input("candidate response: ")
        start_computing_time = time.time()
        raw_feat = get(feat_objects, dim, article, context, candidate)
        logger.info("computed all features in %s seconds" % (time.time() - start_computing_time))
        logger.info("features dim: %s" % (raw_feat.shape,))
        assert dim == len(raw_features)

