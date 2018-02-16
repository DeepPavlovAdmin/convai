from tools.embeddings.word2vec import find_closest_words


class EmbeddedSynonyms(object):

    def __init__(self):
        self.cache = {}

    def find_synonyms(self, word, number=5):
        if word in self.cache:
            return self.cache[word]

        self._check_cache_full()
        word2vec_closest = find_closest_words(word, number)
        self.cache[word] = [synonym.lower() for synonym, distance in word2vec_closest]
        return self.cache[word]

    def _check_cache_full(self, limit=1e4):
        if len(self.cache) > limit:
            self.cache = {}
