import codecs
import cPickle
import glob
import json
import os
import random
from collections import Counter

import numpy as np

import config
from tools import progbar, tokenizer
from tools.embeddings import sentence, word2vec


def knn_dataset(name, filter_empty=True, overwrite_vecs=False):
    dialogues = []
    with codecs.open(config.knn_dialogues[name], 'r', 'utf-8') as f:
        for line in f:
            try:
                sents = line.strip().split('\t')
                if all(sents):
                    dialogues.append((sents[0], sents[1]))
            except:
                continue
    if name in ['chatterbot', 'crafted']:
        # Add lowercase questions for small corpora for better matching
        _questions = {q for q, a in dialogues}
        lower = [(q.lower(), a) for (q, a) in dialogues
                 if q.lower() not in _questions]
        dialogues += lower

    vecs = _load_or_precompute_vecs(
        dialogues, lambda (x, y): x, config.knn_vecs[name],
        overwrite_vecs)
    if filter_empty:
        dialogues, vecs = filter_zero_vecs(dialogues, vecs)
    return (dialogues, vecs)


def filter_zero_vecs(dialogue_pairs, utt_vecs):
    assert utt_vecs.shape[0] == len(dialogue_pairs)
    mask = ~np.isclose(np.linalg.norm(utt_vecs, axis=1), 0.0)
    # print('%d sentence vecs are invalid' % (utt_vecs.shape[0] - np.sum(mask)))
    utt_vecs = utt_vecs[mask]
    dialogue_pairs = [
        pair_ for pair_, bool_ in zip(dialogue_pairs, mask) if bool_]
    assert utt_vecs.shape[0] == len(dialogue_pairs)
    return (dialogue_pairs, utt_vecs)


def _load_or_precompute_vecs(data, project_fun, fpath, overwrite):

    suffix = ''
    if config.word2vec_normalize:
        suffix += '.norm'
    if config.knn_apply_idf:
        suffix += '.idf'
    if fpath.endswith('.npy'):
        fpath = fpath[:-4]
    fpath += suffix + '.npy'

    if os.path.isfile(fpath) and not overwrite:
        utt_vecs = np.load(fpath)
    else:
        # Precompute utterance vectors for dialogues
        pbar = progbar.Progbar(len(data), check_every=100)
        utt_vecs = np.zeros((len(data), word2vec.word_vecs().shape[1]))
        for i, q in enumerate(data):
            if i % 100 == 0:
                pbar.print_progress(
                    i, text='Precomputing sentence vectors... ')
            q = project_fun(q)
            utt_vecs[i] = sentence.utt_vec(
                q, ignore_nonalpha=False, ignore_nonascii=False,
                correct_spelling=True, normalize=True)
        np.save(fpath, utt_vecs)

    if config.knn_method == 'ball_tree':
        for i in range(10):
            assert np.isclose(np.linalg.norm(utt_vecs[i]), 0.0) or \
                np.isclose(np.linalg.norm(utt_vecs[i]), 1.0)

    return utt_vecs.astype(config.knn_floatx)


def compute_idf(article_iterator):
    idf = Counter()
    for i, art in enumerate(article_iterator()):
        tokens = tokenizer.tokenize(
            art, lowercase_first_in_sentence=True, correct_spelling=True)
        idf.update([t for t in word2vec.token_iterator(tokens)])
    num_docs = i
    ret = {w: np.log(num_docs / cnt) for w, cnt in idf.items()}

    with open(config.knn_idf, 'wb') as f:
        cPickle.dump(ret, f)


def _sentence_iterator():
    dialogue_pairs = [
        chatterbot_dialogues]
    for dialogues in dialogue_pairs:
        d = dialogues()
        pbar = progbar.Progbar(len(d), check_every=100)
        for i, (q, a) in enumerate(d):
            pbar.print_progress(i)
            yield q
            yield a


def trivia_qa_questions():
    questions = []
    max_len = np.inf
    excluded = 0
    if config.trivia_max_len:
        max_len = config.trivia_max_len
    for qa_file in glob.glob(config.trivia_path + '/*.json'):
        with codecs.open(qa_file, 'r', 'utf8') as f:
            for question in json.load(f):
                if len(question['q']) < max_len:
                    questions.append(question)
                else:
                    excluded += 1
    print("Trivia excluded %d too long questions" % (excluded,))
    return questions


class StoriesHandler:
    _stories = None

    @classmethod
    def load_stories(cls):
        with codecs.open(config.wiki_samples_articles, 'r', 'utf-8') as f:
            cls._stories = [s.strip() for s in f]

    @classmethod
    def get_one(cls):
        if cls._stories is None:
            cls.load_stories()

        return random.choice(cls._stories)
