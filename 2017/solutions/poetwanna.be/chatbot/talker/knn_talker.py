from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist

import config
import data_manager
from base import ResponderRole
from tools import progbar, tfidf
from tools.embeddings import sentence, word2vec


def get_data(dialogue_sets):
    t = progbar.Timer('Initializing %s datasets (%s)...' %
                      (dialogue_sets, config.word2vec_floatx))
    all_dialogues = []
    all_vecs = []
    for name in dialogue_sets:
        pairs, vecs = data_manager.knn_dataset(name, overwrite_vecs=False)
        all_dialogues += pairs
        all_vecs.append(vecs)
    t.tok()
    return all_dialogues, np.vstack(all_vecs)


def compute_idf():
    def sent_iter():
        dialogues, vecs = get_data(['crafted'])
        del vecs
        for (q, a) in dialogues:
            yield q
            yield a
    np.save(config.knn_idf, tfidf.compute_tf_idf(sent_iter))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scale_confidence(x):
    '''Scales confidence\in[0,1]. It drops sharply below .75'''
    conf = x if x >= 0.75 else sigmoid(15 * (x - 0.68))
    return np.power(conf, 1.5)


def build_knn_data_struct(utt_vecs):
    if config.knn_method == 'cdist':
        return utt_vecs
    elif config.knn_method == 'ball_tree':
        import sklearn.neighbors
        t = progbar.Timer('Building BallTree for k-nn...')
        tree = sklearn.neighbors.BallTree(utt_vecs)
        t.tok()
        return tree


class BaseKnnTalker(ResponderRole):
    name = "base_knn_talker"
    default_confidence = 0.1
    data_sets = []

    def __init__(self, has_context=True,
                 user_context_velocity=0.1,
                 bot_context_velocity=0.05,
                 deterministic=False,
                 **kwargs):
        super(BaseKnnTalker, self).__init__(**kwargs)
        assert self.data_sets != []
        tfidf.init_data()
        word2vec.init_data()
        self.dialogue_pairs, _utt_vecs = get_data(self.data_sets)
        self.utt_vecs = build_knn_data_struct(_utt_vecs)

        self.has_context = has_context
        self.user_context_velocity = user_context_velocity
        self.bot_context_velocity = bot_context_velocity
        self.user_context_vec = None
        self.bot_context_vec = None
        self.deterministic = deterministic

    def new_state(self):
        return {'user_context_vec': self.user_context_vec,
                'bot_context_vec': self.bot_context_vec}

    def _update_context(self, utt, context_vec, velocity):
        if not type(utt) is np.ndarray:
            utt = sentence.utt_vec(utt, ignore_nonalpha=False,
                                   correct_spelling=True, normalize=True)
            utt = utt.astype(config.knn_floatx)
        if context_vec is None:
            context_vec = np.zeros_like(utt)
        context_vec += utt
        context_vec *= velocity
        return context_vec

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict,
                    eps=0.01, print_debug=True):
        normalize = True

        if self.has_context and last_user_utt_dict and last_bot_utt:
            state['user_context_vec'] = self._update_context(
                last_user_utt_dict['corefed_utt'],
                state['user_context_vec'],
                self.user_context_velocity)
            state['bot_context_vec'] = self._update_context(
                last_bot_utt, state['bot_context_vec'],
                self.bot_context_velocity)

        user_utt = user_utt_dict['corefed_utt']

        if config.knn_method == 'ball_tree':
            # BallTree uses euclid dist - vectors need to be normalized
            assert normalize
        s_vec = sentence.utt_vec(user_utt, ignore_nonalpha=False,
                                 correct_spelling=True, normalize=normalize,
                                 print_debug=print_debug)
        s_vec = s_vec.astype(config.knn_floatx)
        if self.has_context:
            if state['user_context_vec'] is not None:
                s_vec += state['user_context_vec']
            if state['bot_context_vec'] is not None:
                s_vec += state['bot_context_vec']

        response, confidence = None, None
        ind = None
        if not np.isclose(np.linalg.norm(s_vec), 0.0):
            if config.knn_method == 'ball_tree':
                k = 1 if self.deterministic else 5
                dists, inds = self.utt_vecs.query(s_vec[None, :], k=k)
                # NOTE We select random neighbor regardless of the dist;
                #      cdist selects within a radius
                dists = dists[0]
                inds = inds[0]
                close = np.where(np.abs(dists - dists.min()) <= eps)[0]
                idx = np.random.choice(close)
                ind = inds[idx]
                # Translate euclid dist to cosine dist (vectors are unit len)
                confidence = 1.0 - 0.5 * (dists[idx] ** 2)
                response = self.dialogue_pairs[ind][1]
            else:
                # Return a response best matching the question
                d = cdist(self.utt_vecs, s_vec[None, :], metric='cosine')
                if self.deterministic:
                    ind = np.argmin(d)
                    response = self.dialogue_pairs[ind][1]
                    confidence = 1.0 - np.min(d)
                else:
                    close_vecs = np.where(np.abs(d - d.min()) <= eps)[0]
                    if close_vecs.shape[0] > 0:
                        ind = np.random.choice(close_vecs)
                        response = self.dialogue_pairs[ind][1]
                        confidence = 1.0 - np.min(d)
                    if print_debug:
                        print('  %s: Found %d close vecs' %
                              (self.name, close_vecs.shape[0]))
        if response is None:
            if self.deterministic:
                response = "Nice."
                confidence = self.default_confidence
            else:
                # By default return a random question
                ind = np.random.randint(len(self.dialogue_pairs))
                response = self.dialogue_pairs[ind][0]
                confidence = self.default_confidence

        if print_debug and ind is not None:
            print(('    Q: %s' %
                   (self.dialogue_pairs[ind][0])).encode('utf-8'))
            print(('    A: %s' %
                   (self.dialogue_pairs[ind][1])).encode('utf-8'))

        return state, response, scale_confidence(confidence)


class KnnTalker(BaseKnnTalker):
    name = "knn_talker"
    data_sets = ['crafted']

class ChatterbotKnnTalker(BaseKnnTalker):
    name = "chatterbot_knn_talker"
    data_sets = ['chatterbot']


class CraftedKnnTalker(BaseKnnTalker):
    name = "crafted_knn_talker"
    data_sets = ['crafted']


class ArticleAuxKnnTalker(BaseKnnTalker):
    name = "article_aux_knn_talker"
    data_sets = ['this_article']

    def __init__(self, **kwargs):
        super(ArticleAuxKnnTalker, self).__init__(**kwargs)
        self.has_context = False
