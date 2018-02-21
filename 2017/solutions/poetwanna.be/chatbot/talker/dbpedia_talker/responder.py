from __future__ import division

from .dbpedia import ResourceFinder
from .tokenizer import Tokenizer
from .response_builder import ResponseBuilder
from .property_matcher import PropertyMatcher


class Responder(object):

    def __init__(self):
        self.resource_finder = ResourceFinder()
        self.property_matcher = PropertyMatcher()
        self.response_builder = ResponseBuilder()
        self.tokenizer = Tokenizer()

    def respond_to(self, user_utt):
        tokens = self.tokenizer.tokenize(user_utt)
        resource, resource_conf = self.resource_finder.find_resource_uri(tokens)
        if not resource:
            return "Subject not found", 0.0
        print "Resource:", resource

        word_values, match_conf = self.property_matcher.match_tokens_with_properties(resource, tokens)
        self._print_word_values(word_values)

        if word_values:
            response, resp_conf = self.response_builder.build(resource, word_values)
        else:
            response, resp_conf = self.response_builder.build_description(resource, user_utt)

        confidence = self._calculate_confidence(resource_conf, match_conf, resp_conf)
        return response, confidence

    def _calculate_confidence(self, resource_conf, match_conf, resp_conf):
        return 4./9 * resource_conf + 3./9 * match_conf + 2./9 * resp_conf

    def _print_word_values(self, word_values):
        for word, values in word_values.iteritems():
            print '\nWord: %s' % word
            for synonym, property, value in values:
                print "    pattern '%s' in '%s':" % (synonym, property)
                for v in value:
                    print "        %s" % v
