#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from talker.base import ResponderRole
from scipy.sparse import csr_matrix


class MCRState(object):
    def __init__(self):
        super(MCRState, self).__init__()
        self.context_dense = np.zeros(300, dtype=mcr.floatx)
        self.context_sparse = csr_matrix(
            (1, len(mcr.words)), dtype=mcr.floatx)

        self.used = []
        self.last_answer = []
        self.my_last_answer = '', None
        self.max_idf = 0.

        self.last_user_utt_max_idf = 0.

        # article bias section
        self.article_context_dense = None
        self.article_context_sparse = None
        self.article_max_idf = None
        self.set_article("")

    def set_article(self, article):
        self.article = article
        (dense_art_v, sparse_art_v, max_article_idf
         ) = mcr.summarize_article(self.article)
        self.article_context_dense = \
            dense_art_v * MCRTalker.article_starting_value
        self.article_context_sparse = \
            sparse_art_v * MCRTalker.article_starting_value
        self.article_max_idf = max_article_idf


class MCRTalker(ResponderRole):
    name = "mcr_talker"

    censorship = True
    random = True

    # weights currently depend on how many words
    # the sentence vectorizer understood, not just on alpha
    alpha = .5
    bleu_max_n = 4
    decay = .25
    cand_num = 5
    ai_words_mult = .3

    default_answer = "I would tell you a quote, but I can't think of any."

    article_starting_value = .8  # set this to 0 to disable article bias
    article_decay = .8

    def __init__(self, **kwargs):
        super(MCRTalker, self).__init__(**kwargs)

        global mcr
        import talker.mcr_talker.MCR as mcr

    def new_state(self):
        return MCRState()

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        del last_user_utt_dict  # unused

        ''' Add last_bot_utt to context '''
        if last_bot_utt:
            if self.was_i_last(state.my_last_answer[0], last_bot_utt):
                state.used.append(state.my_last_answer[1])

            seq_bot = mcr.tokenize(last_bot_utt)
            seq_bot_dense = mcr.dense_vec(seq_bot)
            seq_bot_sparse = mcr.sparse_vec(seq_bot)

            state.last_answer = seq_bot

            state.context_dense += seq_bot_dense * self.ai_words_mult
            state.context_sparse += seq_bot_sparse * self.ai_words_mult

            state.max_idf += max(state.last_user_utt_max_idf,
                                 mcr.max_idf(seq_bot)) * (1 - self.decay)

        del last_bot_utt  # unused

        seq_user = mcr.tokenize(user_utt_dict['corefed_utt'])

        seq_user_dense = mcr.dense_vec(seq_user)
        seq_user_sparse = mcr.sparse_vec(seq_user)

        ''' Add current user_utt to context '''
        state.context_dense *= self.decay
        state.context_sparse *= self.decay

        state.context_dense += seq_user_dense
        state.context_sparse += seq_user_sparse

        state.article_max_idf *= self.article_decay
        state.max_idf *= self.decay

        ''' Continue with respond_to '''
        seq_user_max_idf = mcr.max_idf(seq_user)

        idf_context_weight = state.max_idf + \
            seq_user_max_idf * (1 - self.decay)
        idf_article_weight = state.article_max_idf

        seq_dense = state.context_dense * idf_context_weight + \
            state.article_context_dense * idf_article_weight
        seq_sparse = state.context_sparse * idf_context_weight + \
            state.article_context_sparse * idf_article_weight

        sim = mcr.get_sparse_sims(seq_sparse) * self.alpha + \
            mcr.get_dense_sims(seq_dense) * (1 - self.alpha)

        sim[state.used] = 0

        if self.censorship:
            sim[mcr.bad_inds] = 0

        if np.isclose(sim.sum(), 0):
            return state, self.default_answer, 0
        else:
            cand = sim.argpartition(-self.cand_num)[-self.cand_num:]

            if self.random:
                probs = mcr.softmax(sim[cand])
                answer_idx = np.random.choice(self.cand_num, p=probs)
            else:
                answer_idx = sim[cand].argmax()

            answer_idx = cand[answer_idx]
            answer = mcr.quotes[answer_idx]
            state.my_last_answer = answer, answer_idx

        n = min(self.bleu_max_n, len(seq_user),
                len(mcr.tokenized_quotes[answer_idx]))
        weights = np.ones(n) / n
        bleu_score = mcr.bleu([seq_user, state.last_answer],
                              mcr.tokenized_quotes[answer_idx],
                              weights=weights,
                              smoothing_function=mcr.sf.method1)

        idf_bonus = state.max_idf + \
            max(seq_user_max_idf, state.article_max_idf) * (1 - self.decay)
        print(bleu_score, idf_bonus)

        ''' Preparation for context updating in next _respond_to call '''
        state.article_context_dense *= self.article_decay
        state.article_context_sparse *= self.article_decay
        state.last_user_utt_max_idf = seq_user_max_idf

        return (state,
                answer,
                sim[answer_idx] / np.exp(bleu_score) *
                np.log(idf_bonus) / np.log(mcr.idf.max()))

    def set_article(self, state, article_dict):
        state.set_article(article_dict['corefed_text'])
        return state


if __name__ == '__main__':
    mcr_t = MCRTalker()
    state = mcr_t.new_state()
    bot_utt = ""

    while(True):
        print("Say smth")
        user_utt = raw_input().decode('utf-8')
        state, bot_utt, score = mcr_t._respond_to(
            state, "", bot_utt, {'corefed_utt': user_utt})
        print(bot_utt, score)
