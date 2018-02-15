from .thesaurus import Thesaurus
from .embedded import EmbeddedSynonyms
from .my_synonyms import MySynonyms
from utils import to_utf8


class SynonymsBase(object):

    def __init__(self):
        self.thesaurus = Thesaurus()
        self.embeddings = EmbeddedSynonyms()
        self.my_rules = MySynonyms()

    def find_synonyms(self, word, tokens):
        from_embeddings = self.embeddings.find_synonyms(word)
        from_thesaurus = self.thesaurus.find_synonyms(word)
        from_my_rules = self.my_rules.find_synonyms(word, tokens)
        synonyms = set(from_thesaurus + from_embeddings + from_my_rules)
        return self._to_utf_8(synonyms)

    def _to_utf_8(self, words):
        return map(to_utf8, words)
