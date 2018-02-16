from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import codecs
import cPickle as pickle
import nltk
import numpy as np

from collections import Counter
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import config
import data_manager
import talker.base
import utils


def force_unicode(s):
    if isinstance(s, unicode):
        return s
    else:
        return codecs.decode(s, 'utf8')


def draw_sample(l):
    return l[np.random.randint(len(l))]


BTW = [
    "By the way, do you know: %s?",
    "Maybe you'll know how to solve this: %s?",
    "Ever seen that one: %s?",
    "Here's one for you: %s",
    "I know the answer, do you: %s?"
]

GOOD_JOB = [
    "Well done, you nailed it!",
    "I knew you'll now it",
    "Correct, now this will be our little secret!"
]

SOSO_JOB = [
    "Funny, I thought it is %s! Now you ask me!",
    "Nah, it's %s!",
    "Didn't you know it's %s! You could have asked me instead ;)",
]


class Glove(utils.Singleton):
    """Compute Glove embedding by reusing the embeddings from MCR.
    """

    def __init__(self):
        super(Glove, self).__init__()
        from talker.mcr_talker import MCR
        self.glove_vectors = MCR.glove_vectors
        self.glove_words = MCR.glove_words
        self.glove_w_to_i = MCR.glove_w_to_i
        self.unk = self.glove_vectors[-100:, :].mean(0)

    def embed_words(self, words, weights=None, unk=None):
        if unk is None:
            unk = self.unk
        ret = np.zeros_like(self.glove_vectors[0])
        weight_sum = 0
        if weights is None:
            weights = [1] * len(words)
        for word, weight in zip(words, weights):
            word_i = self.glove_w_to_i.get(word, -1)
            if word_i >= 0:
                ret += weight * self.glove_vectors[word_i]
            else:
                ret += weight * unk
            weight_sum += weight
        if np.isclose(weight_sum, 0.0):
            return ret
        return ret / weight_sum


def _tokenize(text):
    text = text.lower()
    return nltk.word_tokenize(text)


class TriviaData(utils.Singleton):

    def _init_wordlist(self):
        word_counts = Counter()
        for q in self.questions:
            word_counts.update(_tokenize(q.get('category', '')))
            word_counts.update(_tokenize(q['q']))
            word_counts.update(_tokenize(' '.join(q['a'])))
        self.word_counts = word_counts
        self.words = word_counts.keys()
        self.w_to_i = dict(((w, i) for i, w in enumerate(word_counts.keys())))

    def _init_matrices(self):
        def _update_mat(mat, doc_i, text):
            for word in _tokenize(text):
                mat[doc_i, self.w_to_i[word]] += 1

        category_mat = sparse.lil_matrix(
            (len(self.questions), len(self.w_to_i)))
        question_mat = sparse.lil_matrix(
            (len(self.questions), len(self.w_to_i)))
        answer_mat = sparse.lil_matrix(
            (len(self.questions), len(self.w_to_i)))

        for doc_i, q in enumerate(self.questions):
            _update_mat(question_mat, doc_i, q['q'])
            _update_mat(category_mat, doc_i, q.get('category', ''))
            _update_mat(answer_mat, doc_i, ' '.join(q['a']))

        self.category_mat = category_mat.tocsr()
        self.question_mat = question_mat.tocsr()
        self.answer_mat = answer_mat.tocsr()

    def _init_tfidf(self):
        mat = self.question_mat + self.answer_mat
        self.tf = mat.copy()
        self.tf.data = np.log(self.tf.data) + 1.0
        self.idf = np.log(mat.shape[0] / np.maximum(1.0, (mat > 0).sum(0)))
        self.tf_idf = self.tf.multiply(sparse.csr_matrix(self.idf))
        # append a zero for unknown words
        self.idf = np.array(np.hstack((self.idf, [[self.idf.mean()]])))

    def _init_question_embeddings(self):
        num_q = len(self.questions)
        dummy_embed = self.embed_text("Hi")
        dense_question_embeddings = np.empty(
            (num_q, dummy_embed[0].shape[0]))
        sparse_question_embeddings = sparse.lil_matrix(
            (num_q, dummy_embed[1].shape[1]))

        for i in xrange(num_q):
            q = self.questions[i]
            embedding = self.embed_text(q['q'] + ' '.join(q['a']))
            dense_question_embeddings[i, :] = embedding[0]
            sparse_question_embeddings[i:i+1, :] = embedding[1]
            if (i % 2000) == 0:
                print("Embedding trivia questions, %d/%d)" % (i, num_q))
        sparse_question_embeddings = sparse_question_embeddings.tocsr()

        self.dense_question_embeddings = dense_question_embeddings
        self.sparse_question_embeddings = sparse_question_embeddings

    def embed_text(self, text):
        words = _tokenize(text)
        unk_id = len(self.w_to_i)
        word_idxs = [self.w_to_i.get(w, unk_id) for w in words]
        idfs = self.idf[0, word_idxs]
        dense_vec = self.glove.embed_words(words, idfs)
        sparse_vec = sparse.coo_matrix(
            ([1.0] * len(idfs), ([0] * len(word_idxs), word_idxs)),
            (1, self.idf.shape[1])
        ).tocsr()
        return dense_vec, sparse_vec

    def question_similarity(self, text_embedding):
        dense_dists = cosine_similarity(
            self.dense_question_embeddings, text_embedding[0][None, :])
        sparse_dists = cosine_similarity(
            self.sparse_question_embeddings, text_embedding[1])
        dist = 0.7 * dense_dists.ravel() + 0.3 * sparse_dists.ravel()
        return dist

    def __init__(self):
        super(TriviaData, self).__init__()
        self.glove = Glove()
        self.questions = data_manager.trivia_qa_questions()
        self._init_wordlist()
        self._init_matrices()
        self._init_tfidf()
        self._init_question_embeddings()


class TriviaState(object):
    __slots__ = ['masked_q', 'potential_questions',
                 'my_last_question_text', 'my_last_question',
                 'my_last_question_score', 'rounds']

    def __init__(self):
        super(TriviaState, self).__init__()
        self.masked_q = []
        self.potential_questions = []
        self.my_last_question_text = "etaoin shrdlu"
        self.my_last_question = None
        self.my_last_question_score = 0
        self.rounds = 0


class TriviaTalker(talker.base.ResponderRole):
    name = "trivia_talker"
    default_confidence = 0.1

    def __init__(self, **kwargs):
        super(TriviaTalker, self).__init__(**kwargs)
        if not config.debug:
            self.trivia_data = TriviaData()
        else:
            try:
                with open(config.trivia_cache_path, 'rb') as f:
                    self.trivia_data = pickle.load(f)
                    print("Loaded trivia pickle from",
                          config.trivia_cache_path)
            except Exception as e:
                print("Error loading trivia pickle:", str(e))
                self.trivia_data = TriviaData()
                with open(config.trivia_cache_path, 'wb') as f:
                    pickle.dump(self.trivia_data, f, protocol=-1)
                    print("Saved trivia pickle to", config.trivia_cache_path)

    def new_state(self):
        return TriviaState()

    def respond_to_reply(self, state, user_utt, bot_utt):
        """Respond when we got to ask the question"""
        state.masked_q.append(state.my_last_question)
        state.potential_questions = [
            (q, c) for q, c in state.potential_questions
            if q not in state.masked_q]

        q = self.trivia_data.questions[state.my_last_question]

        print("responding to:", q)

        user_toks = set(_tokenize(user_utt))
        answer_toks = set(_tokenize(' '.join(q['a'])))

        print(user_toks, answer_toks, user_toks.intersection(answer_toks))

        state.my_last_question = None
        state.my_last_question_score = 0

        if user_toks.intersection(answer_toks):
            # Assume the guy has responded!
            ret = state, draw_sample(GOOD_JOB), 10.0
        else:
            # Assume the guy did not respond!
            ret = state, draw_sample(SOSO_JOB) % ', '.join(q['a']), 10.0
        return ret

    def find_question(self, state, text, n=1):
        """Find up to n questions matching the text"""
        td = self.trivia_data
        user_utt_embedding = td.embed_text(text)
        question_sims = td.question_similarity(user_utt_embedding)
        question_sims[state.masked_q] = -1000

        num_q, = question_sims.shape

        best_question_idxs = np.argpartition(
            question_sims, num_q - n)[-n:]

        return [(bi, question_sims[bi]) for bi in best_question_idxs]

    def reduce_question_list(self, ql):
        qd = {}
        for q, s in ql:
            qd[q] = max(s, qd.get(q, s))
        return sorted(qd.items(), key=lambda x: x[1])

    def respond_to_init(self, state, user_utt, bot_utt):
        """Respond to any generic prompt"""
        td = self.trivia_data

        best_questions = self.find_question(state, user_utt, n=5)
        best_question_idx, confidence = draw_sample(best_questions)

        # decay with time potential_questions???
        state.potential_questions.append((best_question_idx, confidence))
        state.potential_questions = self.reduce_question_list(
            state.potential_questions)
        best_question_idx, confidence = state.potential_questions[-1]

        q = td.questions[best_question_idx]
        prompt = draw_sample(BTW) % q['q']
        print("asking:", q)

        state.my_last_question = best_question_idx
        state.my_last_question_text = prompt
        state.my_last_question_score = confidence
        return state, prompt, confidence

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        del last_user_utt_dict  # unused

        state.rounds += 1
        user_utt = user_utt_dict['corefed_utt'] or ''
        bot_utt = last_bot_utt or ''
        user_utt = force_unicode(user_utt)
        bot_utt = force_unicode(bot_utt)
        print("Trivia responder: ", user_utt, bot_utt)
        if self.was_i_last(state.my_last_question_text, last_bot_utt):
            (state, response, confidence) = self.respond_to_reply(
                state, user_utt, bot_utt)
        else:
            (state, response, confidence) = self.respond_to_init(
                state, user_utt, bot_utt)

        return (state, response, confidence)

    def follow_up(self, state, new_bot_utts):
        new_bot_utt = new_bot_utts[0]['utt']
        # dont followup oneself
        if (self.was_i_last(state.my_last_question_text, new_bot_utt) or
                state.my_last_question_score == 0):
            return state, None, None
        # don't followup long utterances and don't start early
        if len(new_bot_utt) > 40 or state.rounds < 3:
            return state, None, None
        fup_prob = (0.5 * state.my_last_question_score) ** 1.5
        return state, new_bot_utt + ' ' + state.my_last_question_text, fup_prob

    def set_article(self, state, article_dict={}):
        if article_dict:
            sentences = nltk.sent_tokenize(article_dict['text'])
            state.potential_questions = []
            for sentence in sentences:
                state.potential_questions.extend(
                    self.find_question(state, sentence, n=2))
            state.potential_questions = self.reduce_question_list(
                state.potential_questions)[-5:]
            np.random.shuffle(state.potential_questions)
            state.potential_questions = state.potential_questions[-2:]
            print("Initial qs: ",
                  [(s, self.trivia_data.questions[q])
                   for q, s in state.potential_questions])
        return state


if __name__ == "__main__":
    prev = u""
    talker = TriviaTalker()
    talker = TriviaTalker()

    print("Enter article")

    state = talker.new_state()
    state = talker.set_article(state, dict(text=raw_input().decode('utf-8')))
    print("System ready, start talking")
    while True:
        x = raw_input().decode('utf-8')
        state, txt, val = talker._respond_to(state, x, prev)
        prev = txt
        print('[%.2f] %s' % (val, txt))
