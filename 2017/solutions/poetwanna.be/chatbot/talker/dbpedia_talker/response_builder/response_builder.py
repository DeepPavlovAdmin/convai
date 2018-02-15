from __future__ import division
import utils
from tools.tokenizer import levenshtein
from .description_builder import DescriptionBuilder
from .age_question_handler import AgeQuestionHandler


class ResponseBuilder(object):

    def __init__(self):
        self.description_builder = DescriptionBuilder()
        self.age_question_handler = AgeQuestionHandler()

    def build(self, resource, words_values):
        if 'old' in words_values:
            age_answer = self.age_question_handler.handle(resource, words_values['old'])
            if age_answer:
                return age_answer, 0.8

        response, confidence = '', 0.0
        for word, related_words_values in words_values.iteritems():
            sentence, conf = self._build_sentence(resource, word, related_words_values)
            response += sentence
            confidence += conf
        return response + '\n', conf / response.count('.')

    def build_description(self, resource, utt):
        return self.description_builder.build(resource, utt)

    def _build_sentence(self, resource, word, related_words_values):
        # sentence = "{resource} {property}{plural} {verb} {values} (pattern '{synonym}' in '{property}'). "
        sentence_pattern = "{resource} {predicate} {values}. "
        closest_match, dist = self._pick_synonym_closest_to_property_name(related_words_values)
        synonym, prop, values = closest_match
        values = utils.get_top_pagerank_values(values)
        sentence = sentence_pattern.format(
            resource=utils.strip_value(resource),
            predicate=self._build_predicate(prop, values),
            values=utils.values_to_phrase(values),
            # synonym=synonym,
        )
        confidence = 1 - dist / len(closest_match)
        return sentence, confidence

    def _build_predicate(self, prop, values):
        if prop.startswith('is') and prop.endswith('Of'):
            if prop.endswith('ForOf'):
                prop = prop[:-len('Of')]
            return utils.camelcase_to_phrase(prop)
        else:
            return '{property}{plural} {verb}'.format(
                property=utils.camelcase_to_phrase(prop),
                plural=utils.get_plural(values),
                verb=utils.get_verb(values)
            )

    def _pick_synonym_closest_to_property_name(self, related_words_values):
        """related_words_values: [(word synonym, resource prop., prop. value)]"""
        dists = map(lambda v: levenshtein(v[0], v[1]), related_words_values)
        min_index, min_dist = min(enumerate(dists), key=lambda x: x[1])
        return related_words_values[min_index], min_dist
