from ..base import ResponderRole
from .responder import Responder

import config
from tools.misc import question_bonus


USE_IDF_SCALING = config.dbpedia_idf_scaling


class DbpediaTalker(ResponderRole):
    name = "Dbpedia talker"
    default_confidence = 0.1

    def __init__(self, **kwargs):
        super(DbpediaTalker, self).__init__(**kwargs)
        self.responder = Responder()

        if USE_IDF_SCALING:
            global idf_score_modifier
            from tools.idf_bonus import idf_score_modifier

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        del state  # unused
        del last_user_utt_dict  # unused
        del last_bot_utt  # unused
        question = user_utt_dict['corefed_utt']
        response, confidence = self.responder.respond_to(question)

        if USE_IDF_SCALING:
            question_tags = user_utt_dict['corefed_tags']
            score_idf_bonus = idf_score_modifier(question_tags,
                                                 verbs=True, adjs=True)
            confidence *= score_idf_bonus

        confidence += question_bonus(question)

        return None, response, confidence
