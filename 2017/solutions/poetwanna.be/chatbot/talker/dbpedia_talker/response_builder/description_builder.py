import re
import utils
from ..dbpedia import DBPedia


class DescriptionBuilder(object):

    MAX_DESC_SENTENCES = 2
    SHORT_DESC_SENTECES = 1
    DESC_LEN_LIMIT = 100

    def __init__(self):
        self.dbpedia = DBPedia()

    def build(self, resource, user_utt):
        abstract = self.dbpedia.get_value(resource, 'abstract')
        abstract = self._clean_description(' '.join(abstract))
        abstract = self._shorten_description(abstract)
        confidence = self._calculate_condfidence(user_utt, resource)
        return abstract, confidence

    def _clean_description(self, description):
        desc = utils.remove_nested_comments(description)
        return re.sub(' +', ' ', desc)

    def _shorten_description(self, description):
        sentences = map(
            lambda s: s.strip(),
            description.split('.')[:self.MAX_DESC_SENTENCES]
        )
        if sum(map(len, sentences)) > self.DESC_LEN_LIMIT:
            sentences = sentences[:self.SHORT_DESC_SENTECES]
        return '. '.join(sentences) + '.'

    def _calculate_condfidence(self, utt, resource):
        utt = utt.lower()
        if self._is_definition_question(utt):
            return 0.7 + self._str_in_utt(resource, utt) * 0.3
        return 0.3

    def _str_in_utt(self, str, utt):
        return str.lower() in utt.encode('utf-8').replace(' ', '_')

    def _is_definition_question(self, utt):
        pref = ['who is ', 'who was ', 'what is ', 'what was ']
        inf = [' who is ', ' who was ', ' what is ', ' what was ']
        return any([utt.startswith(p) for p in pref]) or any([i in utt for i in inf])
