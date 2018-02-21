from __future__ import division

import pygtrie
import utils
from config import dbpedia_resources
from tools.tokenizer import levenshtein


class ResourceFinder(utils.Singleton):

    def __init__(self):
        self.resources = pygtrie.StringTrie(separator='_')
        print "Loading DBPedia resources..."
        for line in open(dbpedia_resources, 'r'):
            resource_name, pagerank = line.split()
            if resource_name.startswith('Category:'):
                continue
            pagerank = float(pagerank.rstrip())
            lowered_name = resource_name.lower().replace('-', '_')
            self.resources[lowered_name] = (pagerank, resource_name)
        print "Done"

    def find_resource_uri(self, tokens):
        """Returns longest full match or the one with best pagerank and prefix match"""
        phrases = self._generate_phrases(tokens)
        best_uri, best_pagerank, full_match, best_phrase = '', 0.0, False, ''
        for phrase in phrases:
            for uri_lower, (pagerank, uri) in self._get_resources_with_prefix(phrase):
                if phrase.lower() == uri_lower and not (full_match and len(best_uri) > len(uri)):
                    full_match = True
                    best_uri = uri
                    best_pagerank = pagerank
                    best_phrase = phrase
                elif phrase.lower() != uri_lower and pagerank > best_pagerank and not full_match:
                    best_uri = uri
                    best_pagerank = pagerank
                    best_phrase = phrase
        confidence = self._calculate_confidence(tokens, best_uri, best_pagerank, best_phrase)
        return best_uri, confidence

    def get_resource_pagerank(self, name):
        if name.lower() in self.resources:
            return self.resources[name.lower()][0]
        return 0.0

    def _get_resources_with_prefix(self, phrase):
        """Returns list of (resource_name.lower(), (pagerank, resource_name))"""
        try:
            return list(self.resources.iteritems(prefix=phrase.lower()))
        except KeyError:
            return []

    def _generate_phrases(self, tokens):
        phrases = []
        for length in range(1, len(tokens) + 1):
            i = 0
            while i < len(tokens) and i + length <= len(tokens):
                phrases.append('_'.join(tokens[i: i + length]).encode('utf-8'))
                i = i + 1
        return self._filter_phrases(phrases)

    def _filter_phrases(self, phrases):
        dummy_phrases = [
            'who', 'when', 'where', 'what', 'how', 'so', 'as'
            'a', 'an', 'is', 'are', 'was', 'were', 'did',
            'in', 'at', 'on', 'by', 'the', 'of',
            'who_is', 'who_was', 'what_is', 'what_was',
        ]
        return [p for p in phrases if p.lower() not in dummy_phrases and len(p) > 1]

    def _calculate_confidence(self, tokens, uri, pagerank, phrase):
        try:
            # max_pagerank = 12057
            match_len = len(phrase) / len(' '.join(tokens))
            dist = levenshtein(phrase, uri) / len(uri)
            return 0.65 * (1 - dist) + 0.35 * match_len
        except:
            return 0.0
