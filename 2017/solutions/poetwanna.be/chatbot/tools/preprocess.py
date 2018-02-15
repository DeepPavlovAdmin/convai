import utils
import config
import tokenizer
from corenlp import CoreNLPWrapper
from coref_resolver import CoreferenceResolver
from tools.tokenizer import Speller


DEBUG = config.debug
if DEBUG:
    import sys


class Preprocessor(utils.Singleton):

    COREF = 'coref'
    SPELLER = 'speller'

    def __init__(self, corenlp_url):
        self.nlp = CoreNLPWrapper(corenlp_url)
        self.coref = CoreferenceResolver(corenlp_url)

    def new_state(self):
        state = {
            self.COREF: self.coref.new_state(),
            self.SPELLER: Speller()
        }
        return state

    def set_article(self, state, article):
        state[self.SPELLER].add_words_from(article)
        try:
            state[self.COREF], corefed_article = self.coref.set_article(
                state[self.COREF], article)
        except:
            if DEBUG:
                print("error:", sys.exc_info()[0])
            corefed_article = article

        try:
            tagged_article = self.nlp.pos_tag(article, parse=True)
        except:
            if DEBUG:
                print("error:", sys.exc_info()[0])
            tokens = article.split()
            tagged_article = [(w, config.unknown_tag) for w in tokens]

        try:
            tagged_corefed_article = self.nlp.pos_tag(corefed_article,
                                                      parse=True)
        except:
            if DEBUG:
                print("error:", sys.exc_info()[0])
            tokens = corefed_article.split()
            tagged_corefed_article = [(w, config.unknown_tag) for w in tokens]

        article_dict = {
            'text': article,
            'text_tags': tagged_article,
            'corefed_text': corefed_article,
            'corefed_text_tags': tagged_corefed_article
        }

        return state, article_dict

    def preprocess(self, state, user_raw_utt, last_bot_utt):
        if last_bot_utt is not None:
            state[self.COREF] = self.coref.bot_utterance(
                state[self.COREF], last_bot_utt)

        user_raw_utt = user_raw_utt.strip()
        spelled_utt = state[self.SPELLER].spell_sentence(user_raw_utt)

        try:
            spelled_tags = self.nlp.pos_tag(spelled_utt, parse=True)
        except:
            if DEBUG:
                print("error:", sys.exc_info()[0])
            tokens = tokenizer.tokenize(spelled_utt)
            spelled_tags = list(zip(tokens, len(tokens)*[config.unknown_tag]))

        if any(map(lambda (_, tag):
                   tag in self.coref.accepted_tags, spelled_tags)):
            try:
                state['coref'], corefed_utt = self.coref.user_utterance(
                    state['coref'], spelled_utt)
                corefed_tags = self.nlp.pos_tag(corefed_utt, parse=True)
            except:
                if DEBUG:
                    print("error:", sys.exc_info()[0])
                corefed_utt = spelled_utt
                corefed_tags = spelled_tags
        else:
            corefed_utt = spelled_utt
            corefed_tags = spelled_tags

        user_utt_dict = {
            'raw_utt': user_raw_utt,
            'spelled_utt': spelled_utt,
            'spelled_tags': spelled_tags,
            'corefed_utt':  corefed_utt,
            'corefed_tags': corefed_tags
        }

        if DEBUG:
            print(5*'-' + "<DEBUG>" + 5*'-')
            for key in user_utt_dict:
                print(key, user_utt_dict[key])
            print()
            for entry in state['coref']:
                print(entry)
            print(5*'-' + "</DEBUG>" + 5*'-')

        return state, user_utt_dict
