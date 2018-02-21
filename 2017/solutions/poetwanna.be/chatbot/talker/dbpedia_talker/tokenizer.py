from tools import tokenizer
# from tools.embeddings.word2vec import token_iterator


class Tokenizer(object):

    def tokenize(self, sentence):
        tokens = tokenizer.tokenize(sentence)
        cleaned = self._clean_tokens(tokens)
        # grouped = list(token_iterator(cleaned))
        return cleaned

    def _clean_tokens(self, tokens):
        result = []
        for token in tokens:
            for suff in ["'s", "'"]:
                if token.endswith(suff):
                    token = token[:-len(suff)]
            result.append(token.encode('utf-8'))
        return result

    def clean_entities(self, tokens):
        blacklist = [
            'who', 'when', 'where', 'what', 'how', 'so', 'as',
            'a', 'an', 'is', 'are', 'was', 'were', 'did', 'do',
            '?', '!', ',', '.', '\'s', ';',
            'in', 'at', 'on', 'by', 'the', 'of', 'to', 'not',
            'he', 'she', 'it', 'we', 'you', 'they',
            'their', 'our', 'his', 'yours', 'my', 'her', 'its'
        ]
        return [t for t in tokens if t.lower() not in blacklist and len(t) > 1]
