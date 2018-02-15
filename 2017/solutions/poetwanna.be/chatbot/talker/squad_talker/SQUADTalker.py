from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import config
import tools.wikipedia
import async.wikipedia

from time import time
from talker.base import ResponderRole
from tools.misc import question_bonus
from talker.squad_talker.squad_utils import tokenize, \
    tokenize_article, lower_list, parse_question, rephrase_question, \
    unique_pars, add_questionmark


debug = True
two_searches = False
use_swt_penalty = True

USE_IDF_SCALING = config.squad_idf_scaling
SQUAD_NEGATIVE = config.squad_negative
NOT_A_WORD = '<not_a_word>'


model_files = [
    config.squad_models_path + '6B.best.npz',
    config.squad_models_path + '6B.best.neg.npz'
]

model_file = model_files[1] if SQUAD_NEGATIVE else model_files[0]


def lower_if_needed(list_of_lists):
    if config.glove_ver == '840B':
        return list_of_lists
    return map(lower_list, list_of_lists)


class SQUADTalkerState(object):
    __slots__ = [
        'main_article_title',
        'main_article_pars_tokens',
        'main_article_pars_tokens_cased',
        'article_tokens',
        'article_tokens_cased',
        'main_article_score',
    ]

    def __init__(self):
        super(SQUADTalkerState, self).__init__()

        # the root article from Wikipedia (not article zero!)
        self.main_article_title = None
        self.main_article_pars_tokens = []
        self.main_article_pars_tokens_cased = []
        self.main_article_score = 0.

        # article zero
        self.article_tokens = []
        self.article_tokens_cased = []


class SQUADTalker(ResponderRole):

    name = "squad_talker"

    max_wiki_par_num = 30
    beam_size = 1
    default_ans = [("I don't know what you are talking about.", 0)]
    word_cap = 500

    use_main = True

    # num of top pars from main article to check
    max_main_par_num = 3

    # multiplier of score of top pars from main article
    main_article_weight = 1.1

    # applied to pars found by indexer if they are from main article
    main_title_bonus = 1.3

    # fixed weight of article zero
    article_weight = 1.1

    # timeout
    batch_size = 5
    time_max = config.squad_timeout

    def __init__(self, **kwargs):
        super(SQUADTalker, self).__init__(**kwargs)

        from talker.squad_talker.AnswerBot import AnswerBot, AnswerWrapper
        from talker.mcr_talker.MCR import glove_vectors, glove_words

        if self.async:
            self.wiki = async.wikipedia.Wikipedia(config.default_wiki)
        else:
            self.wiki = tools.wikipedia.Wikipedia(config.default_wiki)

        self.answerbots = [AnswerBot(model_file, glove_vectors,
                                     glove_words, negative=SQUAD_NEGATIVE)]

        self.wrapper = AnswerWrapper()

        if USE_IDF_SCALING:
            global idf_score_modifier
            from tools.idf_bonus import idf_score_modifier

        global STOP_WORDS
        from tools.wikipedia.stop_words import stop_words as STOP_WORDS

    def new_state(self):
        return SQUADTalkerState()

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        del last_user_utt_dict  # unused
        del last_bot_utt  # unused
        user_utt = user_utt_dict['corefed_utt']
        user_utt_tags = user_utt_dict['corefed_tags']

        time_from_start = time()

        # if it's not a question, return
        if not parse_question(user_utt_tags):
            return (state,) + self.default_ans[0]

        question = lower_if_needed([tokenize(user_utt)])[0]
        question = add_questionmark(question)

        # Check article first, it's fast

        # find question key words
        # don't automatically add main article if there are no key words in it
        q_key_words = {(w.lower(), t) for (w, t) in user_utt_tags}
        q_key_words = {w for (w, t) in q_key_words if t[:2] in ['NN', 'JJ'] and
                       w not in STOP_WORDS}

        if debug:
            print("SQUAD: key words:", q_key_words)

        par_scr = []
        pars = []
        pars_cased = []
        titles = []

        # adding article zero
        if state.article_tokens:
            if self.on_topic(q_key_words, state.article_tokens):
                pars.append(state.article_tokens)
                pars_cased.append(state.article_tokens_cased)
                par_scr.append(self.article_weight)
                titles.append('Article zero')
            else:
                print("SQUAD: Discarding article zero.")

        # adding top paragraphs from main article
        if state.main_article_title is not None:
            for i in range(len(state.main_article_pars_tokens)):
                if self.on_topic(q_key_words,
                                 state.main_article_pars_tokens[i]):
                    pars.append(state.main_article_pars_tokens[i])
                    pars_cased.append(state.main_article_pars_tokens_cased[i])
                    par_scr.append(
                        self.main_article_weight * state.main_article_score)
                    titles.append('Main article: %s' %
                                  state.main_article_title)
                else:
                    print("SQUAD: Discarding main article par", i)

        ans, scr = self.extract_answers(question, pars, pars_cased)

        start_batch_from = len(pars)

        # finding related paragraphs in Wiki
        if debug:
            t0 = time()

        wiki_pars_rephrased, swt_penalty = self.get_paragraphs(
            state, user_utt_tags)

        if debug:
            print("First  get_paragraphs:", time()-t0)
            t0 = time()

        if two_searches:
            wiki_pars_regular, _ = self.get_paragraphs(
                state, user_utt_tags, rephrase=False)

            if debug:
                print("Second get paragraphs:", time()-t0)
                t0 = time()

            wiki_pars = unique_pars(wiki_pars_regular + wiki_pars_rephrased,
                                    limit=self.max_wiki_par_num)

            if debug:
                print("Unique pars:", time()-t0)
                t0 = time()
        else:
            wiki_pars = wiki_pars_rephrased

        pars_tokenized_cased = [tokenize(par[1]) for par in wiki_pars]
        pars_tokenized = lower_if_needed(pars_tokenized_cased)

        wiki_pars = [(
            wiki_pars[i][0],
            pars_tokenized[i],
            pars_tokenized_cased[i],
            wiki_pars[i][2]) for i in range(len(wiki_pars))]

        if not wiki_pars:
            if not pars:
                return (state,) + self.default_ans[0]
        else:
            par_scr_wiki, pars_wiki, pars_cased_wiki, titles_wiki = \
                map(list, zip(*wiki_pars))

            pars_wiki_order = (-np.array(par_scr_wiki)).argsort()
            pars_wiki = [pars_wiki[i] for i in pars_wiki_order]
            pars_cased_wiki = [pars_cased_wiki[i] for i in pars_wiki_order]
            par_scr_wiki = [par_scr_wiki[i] for i in pars_wiki_order]
            titles_wiki = [titles_wiki[i] for i in pars_wiki_order]

            pars.extend(pars_wiki)
            pars_cased.extend(pars_cased_wiki)
            par_scr.extend(par_scr_wiki)
            titles.extend(titles_wiki)

        if debug:
            print("Pars preprocessing:", time()-t0)
            t0 = time()

        # extracting answers from Wiki paragraphs (with timeout)
        avg_batch_time = 0.0
        idx = start_batch_from
        time_ansbot = time()

        while len(ans) < len(pars) and \
                self.time_max - (time() - time_from_start) > avg_batch_time:
            batch = slice(idx, idx + self.batch_size)
            ans_batch, scr_batch = self.extract_answers(
                question, pars[batch], pars_cased[batch])
            idx += self.batch_size
            ans.extend(ans_batch)
            scr.extend(scr_batch)
            avg_batch_time = (time() - time_ansbot) / (idx / self.batch_size)

        if not ans:
            return (state,) + self.default_ans[0]

        if len(ans) < len(pars):
            pars = pars[:idx]
            pars_cased = pars_cased[:idx]
            par_scr = par_scr[:idx]
            titles = titles[:idx]

        # aggregation of Wiki score and squad score
        total_scr = np.array(par_scr) * np.array(scr)

        # choosing the best answer
        order = (-total_scr).argsort()
        answers = []

        for idx, i in enumerate(order):
            s = total_scr[i]
            a = self.wrapper.wrap(' '.join(ans[i]), s)
            if not SQUAD_NEGATIVE or NOT_A_WORD not in a:
                answers.append((a, s))
            if debug:
                print("\n\nSQuAD: Checking article '%s'" % titles[i])
                print("Paragraph content: '%s'" % ' '.join(pars[i]))
                print("answer:", a, s)

        if debug:
            print('SQUAD: Checked ', len(ans), 'paragraphs before timeout.')

        answers = sorted(answers, key=lambda x: -x[1])

        if debug:
            print("SQuAD negative answers",
                  "enabled." if SQUAD_NEGATIVE else "disabled.")
            if SQUAD_NEGATIVE:
                print("Negative answers for",
                      len(total_scr) - len(answers), "paragraphs.")

        if not answers:
            answers = self.default_ans

        ret_text, ret_confidence = answers[0]
        ret_confidence += question_bonus(user_utt)

        # score modifications
        if USE_IDF_SCALING:
            idf_mod = idf_score_modifier(
                user_utt_tags, verbs=True, adjs=True)
            if debug:
                print(user_utt_tags)
                print('SQuAD idf_score: ', idf_mod)
            ret_confidence *= idf_mod

        if use_swt_penalty:
            ret_confidence *= swt_penalty

        return state, ret_text, ret_confidence

    def set_article(self, state, article_dict):
        article_text = article_dict['corefed_text']
        state.article_tokens_cased = tokenize(article_text)
        state.article_tokens = lower_if_needed([state.article_tokens_cased])[0]

        if self.use_main:
            # we search Wikipedia for the root article
            doc_zero = tokenize_article(article_text)
            doc_zero = [w.encode('utf8') for w in doc_zero]

            sc, res, tit = self.wiki.get_main_article(
                doc_zero, self.max_main_par_num, debug=debug)
            if tit is not None:
                state.main_article_pars_tokens_cased = map(tokenize, res)
                state.main_article_pars_tokens = lower_if_needed(
                    state.main_article_pars_tokens_cased)
                state.main_article_title = tit
                state.main_article_score = sc
                if debug:
                    print('SQuAD: Found main article:', tit)
            elif debug:
                print('SQuAD: Main article not found.')
        return state

    def get_paragraphs(self, state, user_utt_tags, rephrase=True):
        t0 = time()
        if rephrase:
            wiki_query, swt_penalty = rephrase_question(user_utt_tags)
        else:
            wiki_query = [t[0] for t in user_utt_tags]
            swt_penalty = 1.

        wiki_query = lower_list(wiki_query)
        wiki_query = [w.encode('utf8') for w in wiki_query]

        if debug:
            print("    get_pars from start:", time()-t0)
            t0 = time()

        wiki_pars = self.wiki.find_paragraphs(
            wiki_query,
            self.max_wiki_par_num,
            state.main_article_title,
            self.main_title_bonus,
            debug=debug)

        if debug:
            print("    get_pars wiki search:", time()-t0)
            t0 = time()

        wiki_pars = [par for par in wiki_pars if len(
            par[1].split()) <= self.word_cap]

        return wiki_pars, swt_penalty

    def on_topic(self, q_key_words, par_toks):
        par_low_toks = set(lower_list(par_toks))
        return bool(q_key_words & par_low_toks)

    def extract_answers(self, question, paragraphs, paragraphs_cased):
        ans_tok = [[]] * len(paragraphs)
        ans_scr = [0] * len(paragraphs)

        if not paragraphs:
            return ans_tok, ans_scr

        for i in range(len(self.answerbots)):
            ansi, scri = zip(*self.answerbots[i].get_answers(
                question, paragraphs, paragraphs_cased,
                beam=self.beam_size))
            for j in range(len(ansi)):
                a = ansi[j]
                s = scri[j]
                if s > ans_scr[j]:
                    ans_tok[j] = a
                    ans_scr[j] = s
        return ans_tok, ans_scr
