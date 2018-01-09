import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize

import string

class IntentClassifier:
    """
    IntentClassfier is a knn classifier on mean word embeddings of words in sentence
    """
    def __init__(self, path_to_datafile='./data/data.tsv', path_to_embedding='./data/glove.6B.100d.txt'):
        """
        Loads word embeddings and computes sentence mean word embedding vector

        Args:
            path_to_datafile: tsv file with classes and sentences
            path_to_embedding: embeddings
        """
        print('Loading embeddings: {}'.format(path_to_embedding))
        self.embeddings = None
        with open(path_to_embedding, 'r') as fin:
            self.embeddings = {line.split(' ')[0]: np.array(list(map(lambda x: float(x), line.split(' ')[1:]))) for line in fin}
        assert self.embeddings is not None

        # normalize
        for w in self.embeddings:
            self.embeddings[w] = self.embeddings[w] / np.linalg.norm(self.embeddings[w])

        self.stopwords = set(stopwords.words('english')) - set(['about'])
        print('Loading examples from datafile: {}'.format(path_to_datafile))
        # class_id -> list of examples
        self.data = dict()
        with open(path_to_datafile, 'r') as fin:
            for line in fin:
                cl, sent = line.strip().split('\t')
                sent = {
                    'text': sent,
                    'emb': self._sent_to_emb(sent),
                }
                if cl in self.data:
                    self.data[cl].append(sent)
                else:
                    self.data[cl] = [sent]
        # class embedding is mean embedding of all sentences in class
        self.class_embds = {cl: np.mean(list(map(lambda x: x['emb'], self.data[cl])), axis=0) for cl in self.data}

    def _sent_to_emb(self, sent):
        """
        computes mean word embedding of sentence filtering punctuation and stop words
        Args:
            sent: sentence

        Returns:
            mean word embedding
        """
        sent = ''.join(filter(lambda x: x not in string.punctuation, sent.lower()))
        words = list(filter(lambda x: x not in self.stopwords, word_tokenize(sent)))
        embds = [self.embeddings[w] for w in words if w in self.embeddings]
        if len(embds) == 0:
            return np.zeros_like(self.embeddings['test'])
        return np.mean(np.stack(embds), axis=0)

    def _cosine_distance(self, a, b):
        if np.linalg.norm(a) * np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_scores(self, sent):
        res = {cl: self._cosine_distance(self.class_embds[cl], self._sent_to_emb(sent)) for cl in self.class_embds}
        for cl in self.data:
            m = 0
            for cl_s in self.data[cl]:
                score = self._cosine_distance(cl_s['emb'], self._sent_to_emb(sent))
                print(cl_s['text'], score)
                m = max(m, score)
            res[cl + '_max'] = m
        return res

    def score(self, a, b):
        return self._cosine_distance(self._sent_to_emb(a), self._sent_to_emb(b))

    def knn(self, sent, k=5):
        """
        Args:
            sent: sentence
            k: number of nearest neighbours for knn

        Returns:
            max_class and max_score for max_class
        """
        assert k % 2 != 0
        sent_embedded = self._sent_to_emb(sent)
        scores = []
        for cl in self.data:
            for cl_s in self.data[cl]:
                score = self._cosine_distance(cl_s['emb'], sent_embedded)
                scores.append((cl, cl_s['text'], score))
        scores = list(sorted(scores, key=lambda x: x[2], reverse=True))
        votes_count = {cl: len(list(filter(lambda x: x[0]==cl, scores[:k]))) for cl in self.data}
        max_class, max_votes = None, 0
        for cl in votes_count:
            if max_votes < votes_count[cl]:
                max_votes = votes_count[cl]
                max_class = cl
        return max_class, max(map(lambda x: x[2], filter(lambda x: x[0]==max_class, scores)))
