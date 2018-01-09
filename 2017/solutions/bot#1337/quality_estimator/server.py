import torch
from torch.autograd import Variable

from flask import Flask, jsonify, request
from data_preparation import normalize_words_in_text, base_main, make_vectored_dialogs, get_sentences_matrix

from train_model import convert_to_torch_format, forward_pass
from train_model_sent import forward_pass_sent
from models import DialogModel
from models import UtteranceModel


dialog_model = DialogModel.load('data/models/dialog/model.pytorch')
utterance_model = UtteranceModel.load('data/models/sentence/model.pytorch')


print(dialog_model)
print(utterance_model)


app = Flask(__name__)
user_bot_ix, current_ix, word_ix, _, _ = base_main()


@app.route('/dialog_quality')
def dialog_quality():
    data = request.json

    dialogs = convert_to_dialog_quality_format(data, word_ix, user_bot_ix)
    dialogs = convert_to_torch_format(dialogs)

    assert len(dialogs) == 1

    dialog = dialogs[0]
    out = forward_pass(dialog_model, dialog)
    _, top_i = out.data.topk(1)
    label = top_i[0][0]
    return jsonify(quality_label=label)


@app.route('/utterance_quality')
def utterance_quality():
    data = request.json
    sent_mats = convert_to_utterance_quality_format(data, word_ix, user_bot_ix, current_ix)

    assert(len(sent_mats) == 1)

    sent_mat = Variable(torch.LongTensor(sent_mats))

    out = forward_pass_sent(utterance_model, sent_mat)

    _, top_i = out.data.topk(1)
    label = top_i.resize_(top_i.size()[0]).tolist()
    assert len(label) == 1

    label = label[0]

    return jsonify(quality_label=label)


def convert_to_dialog_quality_format(data, word_ix, user_bot_ix):
    sents = [('<SOD>', ['<SOD>'])]
    for row in data['thread']:
        normalized = normalize_words_in_text(row['text'])
        sents.append((row['userId'], normalized))
    sents.append(('<EOD>', ['<EOD>']))
    print(sents)
    return make_vectored_dialogs([sents], word_ix, user_bot_ix)


def convert_to_utterance_quality_format(data, word_ix, user_bot_ix, current_ix):
    sents = [('<SOD>', ['<SOD>'])]
    for row in data['thread']:
        normalized = normalize_words_in_text(row['text'])
        sents.append((row['userId'], normalized))

    sent_context = sents[-5:]
    cur_sent = normalize_words_in_text(data['current']['text'])
    cur_user = data['current']['userId']
    sent = (cur_user, cur_sent)

    sent_row = (sent_context, sent, -1)
    sent_mats, _ = get_sentences_matrix([sent_row], word_ix, user_bot_ix, current_ix)
    return sent_mats


if __name__ == '__main__':
    app.run(debug=True)
