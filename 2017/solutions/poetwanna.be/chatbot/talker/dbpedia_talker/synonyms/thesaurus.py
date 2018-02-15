import pygtrie
from os import path
from config import thesaurus_path

IDX_PATH = path.join(thesaurus_path, 'th_en_US_v2.idx')
DAT_PATH = path.join(thesaurus_path, 'th_en_US_v2.dat')


class Thesaurus(object):

    def __init__(self):
        self.word_offset = pygtrie.StringTrie(separator=' ')
        print "Loading Thesaurus..."
        for line in open(IDX_PATH, 'r'):
            if '|' in line:
                word, offset = line.split('|')
                offset = int(offset)
                self.word_offset[word] = offset
        print "Done"
        self.cache = {}

    def find_synonyms(self, word, limit=5):  # word stem?
        if word in self.cache:
            return self.cache[word]

        self._check_cache_full()
        if word in self.word_offset:
            with open(DAT_PATH, 'r') as f:
                f.seek(self.word_offset[word], 0)
                f.readline()
                self.cache[word] = self._line_to_words(f.readline())[:limit]
                return self.cache[word]
        return []

    def _line_to_words(self, line):
        result = []
        for word in line.split('|')[1:]:
            if '(antonym)' not in word:
                word = word.split('(generic term)')[0]
                word = word.split('(similar term)')[0]
                result.append(word.strip())
        return result

    def _check_cache_full(self, limit=1e4):
        if len(self.cache) > limit:
            self.cache = {}
