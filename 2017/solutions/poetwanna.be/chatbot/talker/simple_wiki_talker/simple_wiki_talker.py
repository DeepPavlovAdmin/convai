from __future__ import absolute_import
from __future__ import print_function

import config
import random

from tools.misc import question_bonus
from talker.base import ResponderRole
from tools.simple_wikipedia.title_finder import TitleFinder
from utils import U

from tools.definitions_parser import is_definition, question_simplify, \
    has_pronoun, delete_def_prefix
from talker.simple_wiki_talker.swt_utils import follow_up_text, topic_intro, \
    strange_topic, lower_string, title_key, key_title, word_tokenize, \
    greetings, popular, ZeroDict
from talker.simple_wiki_talker.swt_data import load_data, get_definition, \
    get_wiktionary_definition


GREETING_PROBABILITY = 0.9

USE_IDF_SCALING = config.swt_idf_scaling

DEFINITION_BONUS = 0.  # was 0.2, this is unfair advantage over DbpediaTalker
FOLLOW_UP_BONUS = 0.1
PRONOUN_PENALTY = 0.08
FOLLOW_UP_THR = 0.5
WIKTIONARY_DISCOUNT = 0.9


# Keys in Wikipedia indexer and SWT data structures are strings.
# That's why there is lower_string all over the place in SWT code.


class SimpleWikiTalkerState(object):
    __slots__ = ['followers', 'previous_answer', 'previous_title',
                 'previous_followup', 'what_i_was_defining', 'previous_word']

    def __init__(self):
        self.followers = set()
        self.previous_answer = None
        self.previous_title = None
        self.previous_followup = None
        self.what_i_was_defining = ZeroDict()
        self.previous_word = None


class SimpleWikiTalker(ResponderRole):
    name = "swt_talker"

    def_resp = "I am sorry, I have nothing to say."
    dont_want = "I do not want to define: "

    def __init__(self, **kwargs):
        super(SimpleWikiTalker, self).__init__(**kwargs)
        load_data()
        self.TA = TitleFinder()

        global follow_ups
        from talker.simple_wiki_talker.title_similarity import follow_ups

        if USE_IDF_SCALING:
            global idf_score_modifier
            from tools.idf_bonus import idf_score_modifier

    def new_state(self):
        return SimpleWikiTalkerState()

    def follow_up_continuation(self, state, query_words):
        if not query_words or state.previous_followup is None:
            return ''

        for p in state.previous_followup:
            pt = ' '.join(word_tokenize(p))
            pad = lambda s: ' ' + s + ' '
            if pad(pt) in pad(' '.join(query_words)):
                return p

        for p in state.previous_followup:
            ps = set(word_tokenize(lower_string(p)))
            if ps.issubset(set(query_words)):
                return p
        return ''

    def defult_answer_update_state(self, state):
        state.previous_answer = None
        state.previous_followup = None
        state.previous_title = None
        state.previous_word = None

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        del last_user_utt_dict  # unused

        question = user_utt_dict['corefed_utt']
        original_question = question
        question = question.strip()

        # Precalculate score modifiers
        q_bonus = question_bonus(original_question)

        if USE_IDF_SCALING:
            question_tags = user_utt_dict['corefed_tags']
            score_idf_bonus = idf_score_modifier(question_tags)
            print('SWT idf_score: ', score_idf_bonus)

        if is_definition(question.lower()):
            def_bonus = DEFINITION_BONUS
        else:
            def_bonus = 0.0

        question = delete_def_prefix(question.lower())
        wiktionary_query = question

        # This doesn't do very much, vast majority of these scenarios
        # is solved by idf scaling.
        if ((question in greetings or question in popular)
                and random.random() < GREETING_PROBABILITY):
            self.defult_answer_update_state(state)
            return state, U(self.dont_want + question), 0.01

        query_words = word_tokenize(lower_string(question))

        iwaslast = self.was_i_last(state.previous_answer, last_bot_utt)

        # If SWT talked, remember what it said
        if iwaslast:
            self.TA.ban(state.previous_title)
            state.what_i_was_defining[state.previous_word] += 1
            print('Increasing', state.previous_word)

        # Check if user chose one of the followup topics
        if iwaslast:
            chosen_continuation = self.follow_up_continuation(
                state, query_words)
        else:
            chosen_continuation = ''

        # If no followup was chosen, search Wiki for a phrase to define
        if not chosen_continuation:
            vr = self.TA.query(query_words)

            if not vr:
                response_value = 0
                to_define = None
            else:
                response_value, to_define = vr

            response_value = max(response_value, 0.)**0.5

            # response_value is now approximately in [0,1]
            response_value /= 4.0
            response_value = min(response_value, 1.)
            response_value *= (1 - DEFINITION_BONUS - 2 * FOLLOW_UP_BONUS)
            continuation_multiplier = 1.0
        # Otherwise define chosen followup
        else:
            response_value = 1 - 2 * FOLLOW_UP_BONUS
            to_define = chosen_continuation
            continuation_multiplier = 4.0

        def apply_score_mods(sc):
            if USE_IDF_SCALING:
                sc *= score_idf_bonus
            sc += q_bonus
            sc *= continuation_multiplier
            return sc

        # Continue only if SWT has something to talk about
        if to_define is None:
            self.defult_answer_update_state(state)
            return state, U(self.def_resp), 0

        state.previous_title = to_define
        txt = get_definition(to_define)

        # Probably not needed, but I'm not sure. It doesn't hurt -- Maciek
        if not txt:
            self.defult_answer_update_state(state)
            return state, U(strange_topic(to_define)), 0.1

        if has_pronoun(question):
            response_value -= PRONOUN_PENALTY

        # Multiplier based on how many important words from the query
        # appear in the answer.
        query_words_set = set(query_words)
        answer_words_set = set(
            word_tokenize(lower_string(txt + ' ' + to_define)))

        total_mult = (
            (0.1 + self.TA.idf_sum(query_words_set & answer_words_set)) /
            (0.1 + self.TA.idf_sum(query_words_set)))

        response_value *= total_mult**0.5
        print('First response value=', response_value)

        resp_key = title_key(lower_string(to_define))
        state.followers.add(resp_key)

        # If the match is good enough, continue with it and try to find
        # followup topics.
        if response_value > 0.55:
            new_topics = follow_ups(resp_key)
        # Otherwise try to get wiktionary definition
        else:
            new_topics = []

            wiktionary_answer = get_wiktionary_definition(wiktionary_query)

            if wiktionary_answer:
                wiktionary_answer = U(wiktionary_answer)
                state.previous_word = wiktionary_query
                state.previous_answer = wiktionary_answer

                num_defs = state.what_i_was_defining[wiktionary_query]
                value = (0.9 + def_bonus) * (WIKTIONARY_DISCOUNT**num_defs)

                if len(original_question.split()) == 1:
                    length = len(original_question)
                    if length <= 3:
                        value *= 0.5
                    elif length <= 5:
                        value *= 0.7

                return state, wiktionary_answer, apply_score_mods(value)
            else:
                state.previous_word = None

        # Currently no followups are allowed after Wiktionary answer

        # If base answer is good enough and we have some followups, add them
        if new_topics and total_mult > FOLLOW_UP_THR:
            candidate_pairs = [
                (val, title) for (val, title) in new_topics[1:]
                if (':' not in title and title not in state.followers)][:2]

            subjects = [title for (val, title) in candidate_pairs]

            val = sum([val * FOLLOW_UP_BONUS
                       for (val, title) in candidate_pairs])

            state.followers.update(subjects)

            if len(subjects) == 1:  # quite impossible
                txt += ' You will probably be interested in ' + subjects[0]
            elif len(subjects) > 1:
                txt += ' ' + follow_up_text(subjects[0], subjects[1])
                response_value += val

            state.previous_followup = [key_title(s) for s in subjects]
        else:
            state.previous_followup = None

        txt = U(txt)
        state.previous_answer = txt

        # modify and normalize score
        response_value += def_bonus
        response_value = min(1., max(0., response_value))

        return state, txt, apply_score_mods(response_value)


if __name__ == "__main__":
    print("SWT testing")
    SWT = SimpleWikiTalker()
    state = SWT.new_state()
    prev = "unknown_history"
    prev_x = "unknown_history"

    while True:
        x = raw_input('Me: ')
        x = unicode(x, 'utf8')
        state, txt, val = SWT._respond_to(
            state, prev_x, prev, {'corefed_utt': x, 'corefed_tags': []})
        prev = txt
        print('Bot: %s [%.2f]' % (txt.encode('utf8'), val))
