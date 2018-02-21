from __future__ import print_function

import sys
import config
import random
import talker
import tools.wikipedia
import async.wikipedia

from os.path import join
from talker.base import ResponderRole
from utils import U

from talker.topic_guess_talker.tgt_utils import word_tokenize, phrase_intro, \
    cool_noun_phrases, postproc, wiki_intro, summary_intro


TAG_WIKI_TITLE = 'wiki'
TAG_NOUN_PHRASE = 'np'
TAG_SUMMARY = 'sum'

INTROS = {
    TAG_WIKI_TITLE: wiki_intro,
    TAG_NOUN_PHRASE: phrase_intro,
    TAG_SUMMARY: summary_intro,
}


class TopicGuessTalkerState(object):
    __slots__ = ['cool_things', 'number_it_was_active', 'user_utt']

    def __init__(self):
        super(TopicGuessTalkerState, self).__init__()
        self.cool_things = None
        self.number_it_was_active = 0
        self.user_utt = None


class TopicGuessTalker(ResponderRole):
    name = "topic_guess_talker"

    def_no_idea = u'Sorry, I have no idea what this article is about...'
    def_all = lambda self: random.choice([
        u"I already told you all I got from this text :(",
        u"I think we covered that snippet in it's entirety.",
        u"I don't see anything else of note in the text.",
    ])

    max_n_wiki_titles = 1
    max_n_np = 5  # it can be more than this when top scores are the same
    max_n_summary = 1

    wiki_title_thresh = 0.1
    decay = 0.9
    def_conf = 11.

    def __init__(self, **kwargs):
        super(TopicGuessTalker, self).__init__(**kwargs)

        if self.async:
            self.SE = async.wikipedia.Wikipedia(config.default_wiki)
        else:
            self.SE = tools.wikipedia.Wikipedia(config.default_wiki)

        from tools.summarize import FrequencySummarizer
        self.summarize = FrequencySummarizer().summarize

    def new_state(self):
        return TopicGuessTalkerState()

    def get_wiki_titles(self, article_text):
        doc_zero = word_tokenize(article_text)
        doc_zero = [w.encode('utf8') for w in doc_zero]
        res = self.SE.find_titles(doc_zero, self.max_n_wiki_titles,
                                  debug=True)
        return [x for x in res if x[0] > self.wiki_title_thresh]

    def set_article(self, state, article_dict):
        article = U(article_dict['text'])
        article_tags = U(article_dict['text_tags'])
        article_coref = U(article_dict['corefed_text'])
        article_coref_tags = U(article_dict['corefed_text_tags'])

        print(article)
        print(article_coref)

        state.number_it_was_active = 0

        cool_things = []

        ''' Find cool things about the article '''

        # wikipedia titles
        wiki_titles = self.get_wiki_titles(article_coref)
        cool_things.extend([[sc, tit, TAG_WIKI_TITLE]
                            for (sc, tit) in wiki_titles])

        # most frequent noun phrases
        noun_phrases = cool_noun_phrases(article_coref_tags, self.max_n_np)
        cool_things.extend([[sc, p, TAG_NOUN_PHRASE]
                            for (sc, p) in noun_phrases])

        # summarization
        summary_sents = self.summarize(article, self.max_n_summary)
        cool_things.extend([[sc, sent, TAG_SUMMARY]
                            for (sc, sent) in summary_sents])

        state.cool_things = sorted(cool_things, reverse=True)

        for x in state.cool_things:
            print(x)

        return state

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        state.user_utt = user_utt_dict['spelled_utt']
        return state, None, None

    def follow_up(self, state, new_bot_utts):
        new_bot_utt = new_bot_utts[0]

        if new_bot_utt['talker_name'] != talker.ArticleAuxKnnTalker.__name__:
            return state, None, None

        if not state.cool_things:
            if state.number_it_was_active == 0:
                return state, self.def_no_idea, self.def_conf
            return state, self.def_all(), self.def_conf

        _, cool_txt, cool_tag = state.cool_things.pop(0)
        for i in range(len(state.cool_things)):
            if state.cool_things[i][2] == cool_tag:
                state.cool_things[i][0] *= self.decay
        state.cool_things.sort(reverse=True)

        state.number_it_was_active += 1

        ans_txt = postproc(U(INTROS[cool_tag]() % cool_txt))

        return state, ans_txt, self.def_conf
