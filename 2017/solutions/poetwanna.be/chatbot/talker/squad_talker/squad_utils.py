from __future__ import absolute_import
from __future__ import print_function

from nltk.tokenize import regexp_tokenize
from pattern.en import verbs as vb
from collections import defaultdict as dd
from tools.chunker import parse_tree
from string import punctuation

import config


SWT_QUESTION_PENALTY = .2

punc = set(punctuation)

question_words = set(
    "what where when who how why which whose "
    "what's where's when's who's how's".split())

pronouns = set(
    "i me you he him she her it we they them "  # us = United States
    "this those these "
    "myself yourself himself herself itself ourselves themselves yourselves "
    "my your her his its their our "
    "mine yours hers ours theirs "
    "i'm you're he's she's it's we're they're".split())

question_words_single = set("what where when who how why which whose".split())

aux_vbs = {'do', 'have', 'be'}

swt_starts = {"what is", "what 's", "what was", "what are", "what were",
              "who is", "who 's", "who was", "who are", "who were"}


# Split on whitespaces (excluding) and non-alphanumeric chars (including).
def tokenize(s):
    if type(s) is not unicode:
        s = s.decode('utf8')
    return regexp_tokenize(s, pattern='[^\W_]+|\S')


def tokenize_article(s):
    if not isinstance(s, unicode):
        s = unicode(s, 'utf8')
    r = []
    s = s.strip()
    for a in s:
        if a in punc:
            pass
        else:
            r.append(a)
    return ''.join(r).split()


def lower_list(xs):
    return [x.lower() for x in xs]


def unique_pars(pars, limit, fn=max):
    assert type(limit) is int
    res = dd(lambda: [])
    for s, p, t in pars:
        res[(p, t)].append(s)
    all_pars = [[fn(v), k[0], k[1]] for (k, v) in res.items()]
    return sorted(all_pars, reverse=True)[:limit]


def add_questionmark(q_tokens):
    if q_tokens[-1] not in '!?.':
        q_tokens = q_tokens + ['?']
    return q_tokens


def get_noun_phrases(tree):
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()


def parse_question(utt_tags):
    if not utt_tags:
        return 0
    utt_tags = [(w.lower(), t) for (w, t) in utt_tags]
    qws = [i for i in range(len(utt_tags)) if utt_tags[i][0] in question_words]
    if not qws:
        return 0
    qw_idx = min(qws)
    prons = [i for i in range(qw_idx + 1, len(utt_tags))
             if utt_tags[i][0] in pronouns]
    if prons:
        return 0
    verbs = [i for i in range(qw_idx + 1, len(utt_tags))
             if utt_tags[i][1].startswith('VB')]
    if not verbs:
        return 0
    return 1


def rephrase_question(q_tagged, debug=False):
    score_mod = 1.

    q_tokens = [t[0] for t in q_tagged]
    if q_tagged and q_tagged[0][1] == config.unknown_tag:
        return q_tokens, score_mod

    qws = [i for i in range(len(q_tokens)) if q_tokens[
        i].lower() in question_words_single]

    if not qws:
        if q_tokens[-1] in '!?.':
            q_tokens = q_tokens[:-1]
        return q_tokens, score_mod

    qw = min(qws)
    if q_tokens[-1] not in '!?.':
        q_tokens = q_tokens + ['?']
        q_tagged.append((u'?', u'.'))

    query_beg, query = q_tokens[
        :qw + 1], q_tokens[qw + 1:]

    pos = q_tagged[qw + 1:]

    verbs = [i for i in range(len(query)) if pos[i][1][:2] in ['VB', 'MD']]
    if not verbs:
        return query_beg + query[:-1], score_mod
    verb_idx = min(verbs)

    nouns = [i for i in range(verb_idx+1, len(query))
             if pos[i][1].startswith('NN')]
    if not nouns:
        return query_beg + query[:-1], score_mod

    tree = parse_tree(pos[verb_idx:])
    noun_phrase = list(zip(*get_noun_phrases(tree).next())[0])

    verb_insert_idx = min(i for i in range(len(query)) if query[
                          i:i+len(noun_phrase)] == noun_phrase)
    verb_insert_idx += len(noun_phrase)

    verb = query[verb_idx]

    if pos[verb_idx][1] != 'MD' and vb.conjugate(verb) not in aux_vbs:
        return query_beg + query[:-1], score_mod

    if verb == 'do':
        query_new = query[:verb_idx] + query[verb_idx+1:-1]
    elif verb in ['does', 'did']:
        verb_to_correct_idx = min(
            [i for i in verbs if i >= verb_insert_idx] or [None])
        if verb_to_correct_idx is None:
            query_new = query[:-1]
        else:
            v = query[verb_to_correct_idx]
            corrected = vb.conjugate(
                v, '3sg') if verb == 'does' else vb.conjugate(v, 'ppl')
            query_new = query[:verb_idx] + \
                query[verb_idx+1:verb_to_correct_idx] + \
                [corrected] + query[verb_to_correct_idx+1:-1]
    else:
        query_new = query[:verb_idx] + \
            query[verb_idx+1:verb_insert_idx] + \
            [verb] + query[verb_insert_idx:-1]

    if debug:
        print(q_tokens, qw, verb_insert_idx, query)

    # penalty for questions better suited for SimpleWikiTalker
    if ' '.join(q_tokens[qw:qw+2]).lower() in swt_starts and \
            verb_insert_idx == len(query) - 1:
        print("SQUAD: penalizing swt start")
        score_mod = SWT_QUESTION_PENALTY

    return query_beg + query_new, score_mod
