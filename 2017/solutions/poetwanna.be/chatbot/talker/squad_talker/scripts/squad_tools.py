from __future__ import print_function

import numpy as np
import os
import json
import io


def load_glove(glove_fname, only_words=False):
    words = ['<unk>']
    if not only_words:
        vecs = []
    with io.open(glove_fname, encoding='utf8') as f:
        for line in f:
            l = line[:-1].split(' ', 1)
            words.append(l[0])
            if not only_words:
                vecs.append(np.fromstring(l[1], sep=' ', dtype=np.float32))
    words.append('<not_a_word>')
    if not only_words:
        vecs = np.vstack(vecs)
        vecs = np.concatenate([
            [vecs.mean(axis=0)],
            vecs,
            np.random.normal(size=(1, 300)).astype(np.float32)])
        return words, vecs
    return words


def load_squad_train(path, negative_paths=[], NAW_token=None, NAW_char=3):
    with open(os.path.join(path, 'train_words.json')) as f:
        train_words = json.load(f)
    with open(os.path.join(path, 'train_char_ascii.json')) as f:
        train_char = json.load(f)
    with open(os.path.join(path, 'train_bin_feats.json')) as f:
        train_bin_feats = json.load(f)

    if not negative_paths:
        print("Only positive samples.")
    else:
        print("Using negative samples.")
        train_words, train_char, train_bin_feats = add_NAW_token(
            [train_words, train_char, train_bin_feats], NAW_token)

        for negative_path in negative_paths:
            with open(os.path.join(
                    path, negative_path, 'train_words.json')) as f:
                train_words_neg = json.load(f)
            with open(os.path.join(
                    path, negative_path, 'train_char_ascii.json')) as f:
                train_char_neg = json.load(f)
            with open(os.path.join(
                    path, negative_path, 'train_bin_feats.json')) as f:
                train_bin_feats_neg = json.load(f)

            train_words += train_words_neg
            train_char += train_char_neg
            train_bin_feats += train_bin_feats_neg

    return train_words, train_char, train_bin_feats


def load_squad_dev(squad_path, preproc_path, lower_raw, make_negative=False,
                   NAW_token=None, NAW_char=3):
    with open(os.path.join(squad_path, 'dev-v1.1.json')) as f:
        json_dev = json.load(f)

    dev_pars_raw = {}
    for par in json_dev['data']:
        for con in par['paragraphs']:
            for q in con['qas']:
                context = con['context']
                if lower_raw:
                    context = context.lower()
                dev_pars_raw[q['id']] = context

    with open(os.path.join(preproc_path, 'dev.json')) as f:
        dev = json.load(f)
    with open(os.path.join(preproc_path, 'dev_words.json')) as f:
        dev_words = json.load(f)
    with open(os.path.join(preproc_path, 'dev_char_ascii.json')) as f:
        dev_char = json.load(f)
    with open(os.path.join(preproc_path, 'dev_bin_feats.json')) as f:
        dev_bin_feats = json.load(f)

    if make_negative:
        print("Adding NAW token to dev set.")
        dev, dev_words, dev_char, dev_bin_feats = add_NAW_token(
            [dev, dev_words, dev_char, dev_bin_feats], NAW_token)

    return json_dev, dev_pars_raw, dev, dev_words, dev_char, dev_bin_feats


def add_NAW_token(data, NAW_token, NAW_word=u'<not_a_word>'):
    assert type(NAW_token) == int
    assert len(data) in [3, 4]  # 4 means there is a raw dev

    words, char, bin_feats = data[-3:]

    words_new = [d[:2] + [d[2] + [NAW_token]] for d in words]
    char_new = [[d[0], d[1] + [[1, 3, 2]]] for d in char]
    bin_feats_new = [d + [False] for d in bin_feats]

    result = words_new, char_new, bin_feats_new

    if len(data) == 4:
        raw = data[0]
        raw_new = [d[:2] + [d[2] + [NAW_word]] + [d[3]] for d in raw]
        result = (raw_new,) + result

    return result


def filter_empty_answers(train_data):
    words, char, bin_feats = train_data
    words_new = []
    char_new = []
    bin_feats_new = []

    for i in range(len(words)):
        if words[i][0]:
            words_new.append(words[i])
            char_new.append(char[i])
            bin_feats_new.append(bin_feats[i])

    return words_new, char_new, bin_feats_new


def trim_data(data, trim):
    if trim <= 0:
        return data

    words, char, bin_feats = data

    words_new = []
    char_new = []
    bin_feats_new = []

    for i in range(len(words)):
        if len(words[i][2]) > trim:
            if words[i][0][0][-1] < trim:  # if trimmed par contains answer
                words_new.append(words[i][:2] + [words[i][2][:trim]])
                char_new.append([char[i][0], char[i][1][:trim]])
                bin_feats_new.append(bin_feats[i][:trim])
        else:
            words_new.append(words[i])
            char_new.append(char[i])
            bin_feats_new.append(bin_feats[i])

    return words_new, char_new, bin_feats_new


def train_QANet(net, train_data, model_path, batch_size, num_epochs=20,
                log_interval=200):
    best = 0
    model_filename = os.path.join(model_path, 'model')
    best_name = os.path.join(model_path, '6B.best')
    if net.negative:
        best_name += '.neg'
    for epoch in range(1, num_epochs + 1):
        print('\n\nStarting epoch {}...\n'.format(epoch))
        train_error = net.train_one_epoch(train_data=train_data,
                                          batch_size=batch_size,
                                          log_interval=log_interval)
        print('\n')
        f1 = net.calc_dev_f1(epoch, mode='ep', verbose=False)
        print('\nTraining loss:   {}'.format(train_error))
        print('F1 after epoch %i:' % epoch, f1)

        if np.isnan(train_error):
            print("Encountered NaN, finishing...")
            break

        if f1 > best:
            net.save_params(model_filename + '.ep{:02d}'.format(epoch))
            net.save_params(best_name)
            best = f1
            print('Best F1 so far, model saved.')

    print('Models saved as ' + model_filename)
