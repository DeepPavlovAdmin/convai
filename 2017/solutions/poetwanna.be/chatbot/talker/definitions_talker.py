from __future__ import print_function

from talker.base import ResponderRole
from tools.definitions_parser import is_definition


class DefinitionsTalker(ResponderRole):
    name = 'definitions_talker'

    name_swt = "SimpleWikiTalker"
    name_squad = "SQUADTalker"
    name_dbpedia = "DbpediaTalker"

    thresholds = {
        name_swt: .15,
        name_squad: 0,
        name_dbpedia: .20,
    }
    def_conf = 10

    def __init__(self, **kwargs):
        super(DefinitionsTalker, self).__init__(**kwargs)

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        return user_utt_dict['corefed_utt'], None, None

    def follow_up(self, state, new_bot_utts):
        last_user_utt = state
        if (last_user_utt is None or not is_definition(last_user_utt.lower())):
            return state, None, 0.

        talker_utts = {}
        for d in new_bot_utts:
            tn = d['talker_name']
            if tn in self.thresholds and d['score'] > self.thresholds[tn]:
                talker_utts[tn] = (d['utt'], d['score'])

        if not talker_utts:
            return state, None, 0

        no_response = (None, 0.)
        utt_squad, unused_conf_squad = talker_utts.pop(self.name_squad,
                                                       no_response)
        if not talker_utts:
            return state, utt_squad, self.def_conf

        best = max(talker_utts.keys(), key=lambda k: talker_utts[k][1])
        best_utt, unused_best_sc = talker_utts[best]

        return state, best_utt, self.def_conf
