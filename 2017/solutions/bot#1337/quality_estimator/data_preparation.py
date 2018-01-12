import json
import pickle
import numpy as np
from sys import argv
from nltk import word_tokenize
from collections import Counter, defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def get_label(val):
    if val < 3:
        return 0
    elif val == 3:
        return 1
    elif val > 3:
        return 2


def preserve_good_data(dialogs):
    filtered = []
    for d in dialogs:
        eval1 = d['evaluation'][0]['quality']
        eval2 = d['evaluation'][1]['quality']
        if len(d['thread']) < 4 and (eval1 > 3 or eval2 > 3):
            pass
        elif d['users'][0]['userType'] == 'Human' and d['users'][1]['userType'] == 'Human':
            pass
        elif len(d['thread']) == 0:
            pass
        else:
            filtered.append(d)
    return filtered


def create_dataset(filtered):
    dialogs = []
    labels = []
    for d in filtered:
        context = d['context']
        user_replicas = []
        bot_replicas = []
        if d['users'][0]['userType'] == 'Human' and d['users'][1]['userType'] == 'Bot':
            user = d['users'][0]['id']
            bot = d['users'][1]['id']
        else:
            user = d['users'][1]['id']
            bot = d['users'][0]['id']

        if d['evaluation'][0]['userId'] == bot:
            label = get_label(d['evaluation'][0]['quality'])
        else:
            label = get_label(d['evaluation'][1]['quality'])

        dialog = [('<SOD>', ['<SOD>'])]
        for r in d['thread']:
            words = normalize_words_in_text(r['text'])
            if r['userId'] == user:
                dialog.append(('user', words, r['evaluation']))
            else:
                dialog.append(('bot', words, r['evaluation']))
        dialog.append(('<EOD>', ['<EOD>']))
        dialogs.append(dialog)
        labels.append(label)
    return dialogs, labels


def normalize_words_in_text(text):
    words = [w.lower() for w in word_tokenize(text)]
    words.insert(0, '<BOS>')
    words.append('<EOS>')
    return words


# TODO: Добавить unknown токен!
def make_word_ix(dialogs, start_ix=0):
    word_ix = {}
    vocab = set()
    for d in dialogs:
        for sent in d:
            for w in sent[1]:
                vocab.add(w)
    ix = start_ix
    for w in sorted(vocab):
        word_ix[w] = ix
        ix += 1
    return word_ix


# [ [ [words 1xS], [bots 1xS] ] ]
def make_vectored_dialogs(dialogs, word_ix, user_bot_ix):
    dialogs_vecs = []
    for d in dialogs:
        d_vecs = []
        for sent in d:
            sent_bot_ix = []
            sent_word_ix = []
            for w in sent[1]:
                if w in word_ix:
                    sent_word_ix.append(word_ix[w])
                else:
                    print('WARNING: UNK WORD ({})'.format(w))
                    sent_word_ix.append(0)
                sent_bot_ix.append(user_bot_ix[sent[0]])
            if sent_bot_ix:
                sent_vec = [sent_word_ix, sent_bot_ix]
                d_vecs.append(sent_vec)
        dialogs_vecs.append(d_vecs)
    return dialogs_vecs


def make_dialog_sent_eval_labels(dialogs):
    dialogs_labels = []
    for d in dialogs:
        sent_labels = []
        for s in d:
            if len(s) > 2:
                sent_labels.append(s[2])
            else:
                sent_labels.append(0)
        dialogs_labels.append(sent_labels)
    return dialogs_labels


def oversample(dialogs, labels):
    labels_cnt = Counter(labels)
    grouped = defaultdict(list)
    for ind, label in enumerate(labels):
        grouped[label].append(dialogs[ind])
    grouped[1] = np.repeat(grouped[1], labels_cnt[0] // labels_cnt[1])
    grouped[2] = np.repeat(grouped[2], labels_cnt[0] // labels_cnt[2])
    y_oversampled = np.repeat(0, len(grouped[0]))
    y_oversampled = np.concatenate([y_oversampled, np.repeat(1, len(grouped[1]))])
    y_oversampled = np.concatenate([y_oversampled, np.repeat(2, len(grouped[2]))])

    X_oversampled = np.concatenate([grouped[0], grouped[1], grouped[2]])

    labels_cnt = Counter(y_oversampled)
    print('After oversampling: {}'.format(labels_cnt))

    return X_oversampled, y_oversampled


def create_sentence_evaluation_dataset(dialogs, word_ix, user_bot_ix, current_ix):
    sents = []
    for d in dialogs:
        for ind, sent in enumerate(d):
            if len(sent) > 2 and sent[2] > 0:
                label = sent[2]
                sent_context = d[ind-5:ind]
                sent_row = (sent_context, sent, label)
                sents.append(sent_row)
    sentences_matrix = get_sentences_matrix(sents, word_ix, user_bot_ix, current_ix)
    return sentences_matrix


# [3xSj], 1 - word_id, 2 - user_bot, 3 - current_utt
def get_sentences_matrix(sents, word_ix, user_bot_ix, current_ix):
    sent_mats = []
    labels = []
    for sent_row in sents:
        sent_context, sent, label = sent_row
        labels.append(label)
        sent_mat = get_sent_mat(sent, word_ix, user_bot_ix, current_ix, 'CUR')
        sent_mats_context = [get_sent_mat(sent_c, word_ix, user_bot_ix, current_ix, 'NOT_CUR') for sent_c in sent_context]
        if sent_mats_context:
            sent_mats_context = np.hstack(sent_mats_context)
            sent_mat = np.hstack([sent_mats_context, sent_mat])
        sent_mat_result = np.zeros((3, 50), dtype=np.int64)
        min_shape = min(50, sent_mat.shape[1])
        sent_mat_result[:, :min_shape] = sent_mat[:, :min_shape]
        sent_mats.append(sent_mat_result)
    sent_mats = np.array(sent_mats)
    labels = np.array(labels)
    print(sent_mats.shape, labels.shape)
    return sent_mats, labels
    # padding_idx=0


def get_word_ids(words, word_ix):
    return np.array([word_ix[word] for word in words])


def get_sent_mat(sent, word_ix, user_bot_ix, current_ix, is_current='NOT_CUR'):
    user_bot_id = user_bot_ix[sent[0]]
    current_id = current_ix[is_current]

    words_ids = get_word_ids(sent[1], word_ix)
    user_bot_ids = np.repeat(user_bot_id, len(sent[1]))
    current_ids = np.repeat(current_id, len(sent[1]))

    sent_mat = np.vstack((words_ids, user_bot_ids, current_ids))
    return sent_mat


def base_main():
    with open("data/train_full.json") as f:
        dialogs = json.load(f)

    filtered = preserve_good_data(dialogs)

    dialogs, labels = create_dataset(filtered)

    user_bot_ix = {'user': 1, 'bot': 2, '<SOD>': 3, '<EOD>': 4}
    current_ix = {'NOT_CUR': 1, 'CUR': 2}
    word_ix = make_word_ix(dialogs, 1)

    return user_bot_ix, current_ix, word_ix, dialogs, labels


def main_sent():
    user_bot_ix, current_ix, word_ix, dialogs, labels = base_main()

    sent_mats, labels = create_sentence_evaluation_dataset(dialogs, word_ix, user_bot_ix, current_ix)

    X_train, X_test, y_train, y_test = train_test_split(
        sent_mats, labels, test_size=0.15, random_state=42
    )

    print("X_train: {}; Shape: {}".format(X_train[:2], X_train.shape))
    print("y_train: {}; Shape: {}".format(y_train[:2], y_train.shape))

    with open('data/sent_data.pickle', 'wb') as f:
        pickle.dump([X_train, X_test, y_train, y_test], f)


def main(with_oversampling=False):
    user_bot_ix, _, word_ix, dialogs, labels = base_main()

    dialogs_vectored = make_vectored_dialogs(dialogs, word_ix, user_bot_ix)

    X_train, X_test, y_train, y_test = train_test_split(
        dialogs_vectored, labels, test_size=0.15, random_state=42
    )
    print("X_train: {}".format(X_train[:2]))
    print("y_train: {}".format(y_train[:2]))

    if with_oversampling:
        X_train, y_train = oversample(X_train, y_train)
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

    with open('data/dialogs_and_labels.pickle', 'wb') as f:
        pickle.dump([X_train, X_test, y_train, y_test], f)


if __name__ == '__main__':
    if argv[1] == 'sent':
        main_sent()
    else:
        main()

