import random

from utils import U
from tools.chunker import get_noun_phrases
from collections import defaultdict as dd
from tools.idf_bonus import idf_score_modifier
from string import punctuation
from nltk.corpus import stopwords


punc = set(punctuation)
stopwords = set(stopwords.words('english') + list(punctuation))


def lower(s):
    return U(s).lower()


def add_dot(s):
    if not s or s[-1] not in '.!?':
        s = s + '.'
    return s


def upper_first(s):
    if not s:
        return s
    return s[0].upper() + s[1:]


def postproc(s):
    return add_dot(upper_first(s))


def word_tokenize(s):
    r = []
    s = s.strip()
    for a in s:
        if a not in punc:
            r.append(a)
    return ''.join(r).split()


def phrase_intro():
    return random.choice([
        'The topic of this text is %s',
        'This text is about %s',
        'The text is about %s',
        'The text says something about %s',
        'It is about %s',
        "It's about %s",
        'This article says interesting things about %s',
        'Well, this is an interesting subject: %s',
        "I think it's about %s",
        '%s seems to be the topic of the text.',
        "It looks like it's about %s"
    ])


def wiki_intro():
    return random.choice([
        '%s could be related to the article in some way.',
        'Maybe %s could be relevant to this topic?',
        "I can see some connections to %s",
    ])


def summary_intro():
    return random.choice([
        'This fragment seems particularly important:\n%s',
        'I think this sentence sums it up pretty well:\n%s',
        'The gist of the text is this:\n%s',
    ])


def shorter(title):
    return title


def phrase_score(phrases_counts, phrases_tags, use_idf=True):
    scores = {}
    max_count = float(max(phrases_counts.values()))

    for p in phrases_counts:
        scores[p] = phrases_counts[p] / max_count

    if use_idf:
        for p in phrases_counts:
            scores[p] *= idf_score_modifier(phrases_tags[p],
                                            verbs=True, adjs=True)
    return scores


# returns more than K if >K top scores are the same
def cool_noun_phrases(article_tags, K):
    phrases = get_noun_phrases(article_tags, with_tags=True)
    if not phrases:
        return []

    phrases_tags = {' '.join(zip(*l)[0]): l for l in phrases}

    phrases_counts = dd(int)
    for p in phrases_tags:
        phrases_counts[p] += 1

    scores = phrase_score(phrases_counts, phrases_tags)
    cands = sorted(scores.items(), key=lambda x: -x[1])
    cands = [(sc, p) for (p, sc) in cands]
    max_score = cands[0][0]

    result = []
    for i in range(len(cands)):
        sc, p = cands[i]
        if i < K or sc == max_score:
            result.append((sc, p))
        else:
            break

    return result
