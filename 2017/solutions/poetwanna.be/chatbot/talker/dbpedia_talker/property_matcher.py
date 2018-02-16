from __future__ import division
from collections import defaultdict
from dbpedia import DBPedia
from synonyms import SynonymsBase
from tokenizer import Tokenizer


class PropertyMatcher(object):

    def __init__(self):
        self.dbpedia = DBPedia()
        self.synonyms_base = SynonymsBase()
        self.tokenizer = Tokenizer()

    def match_tokens_with_properties(self, resource, tokens):
        properties = self.dbpedia.get_properties(resource)
        word_values = defaultdict(list)
        filtered_tokens = self._filter_tokens(tokens, resource)
        for word in filtered_tokens:
            if word.lower() in resource.lower():
                continue
            synonyms = self.synonyms_base.find_synonyms(word, tokens)

            for synonym in self.tokenizer.clean_entities(synonyms):
                for prop in properties:
                    if prop == 'sameAs' or prop == 'wasDerivedFrom':
                        continue
                    if self._word_matches_property(synonym, prop):
                        values = self.dbpedia.get_value(resource, prop)
                        word_values[word].append((synonym, prop, values))
        confidence = self._calculate_confidence(filtered_tokens, word_values)
        return word_values, confidence

    def _word_matches_property(self, word, property):
        return word in property.lower()  # + levenshtein?

    def _calculate_confidence(self, tokens, word_values):
        wh_q = any([t.lower() in ['when', 'where'] for t in tokens])
        birth_death_q = any([t.lower() in ['born', 'birth', 'die', 'died'] for t in tokens])
        if wh_q and birth_death_q:
            return 0.9

        if len(tokens) == 0:
            return 1.0

        return len(word_values.keys()) / len(tokens)

    def _filter_tokens(self, tokens, resource):
        return [
            w for w in self.tokenizer.clean_entities(tokens)
            if w.lower() not in resource.lower()
        ]
