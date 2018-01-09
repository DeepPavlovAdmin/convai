import torch
import torch.nn as nn
from torch.autograd import Variable


class DialogModel(nn.Module):
    def __init__(self):
        super(DialogModel, self).__init__()

        # Bx50
        self.word_embeddings = nn.Embedding(11000, 50, padding_idx=0)
        # Bx10
        self.user_bot_embeddings = nn.Embedding(5, 10, padding_idx=0)
        self.rnn = nn.GRU(60, 128, 1)
        self.linear = nn.Linear(128, 3)
        self.softmax = nn.LogSoftmax()

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, 128))

    # input => Bx2xN, B - sentence len
    def forward(self, input, calc_softmax=False):
        word_emb = self.word_embeddings(input[:, 0, :])
        user_bot_emb = self.user_bot_embeddings(input[:, 1, :])
        input_combined = torch.cat((word_emb, user_bot_emb), 2)
        input_combined = input_combined.view(input_combined.size()[1], 1, input_combined.size()[-1])

        rnn_out, self.hidden = self.rnn(input_combined, self.hidden)
        output = self.linear(self.hidden).view(1, 3)

        if calc_softmax:
            probs = self.softmax(output)
            return self.hidden, probs
        else:
            return self.hidden, output

    def save(self, path='data/models/dialog/model.pytorch'):
        torch.save(self, path)
        return True

    @staticmethod
    def load(path='data/models/dialog/model.pytorch'):
        return torch.load(path)


class UtteranceModel(nn.Module):
    def __init__(self):
        super(UtteranceModel, self).__init__()
        # Bx50
        self.word_embeddings = nn.Embedding(11000, 50, padding_idx=0)
        # Bx10
        self.user_bot_embeddings = nn.Embedding(5, 10, padding_idx=0)
        self.cur_embeddings = nn.Embedding(3, 10, padding_idx=0)
        self.rnn = nn.GRU(70, 128, 1, batch_first=True)
        self.linear = nn.Linear(128, 3)
        self.softmax = nn.LogSoftmax()

    # input => Bx2xN, B - sentence len
    def forward(self, input, calc_softmax=False):
        word_emb = self.word_embeddings(input[:, 0, :])
        user_bot_emb = self.user_bot_embeddings(input[:, 1, :])
        cur_emb = self.cur_embeddings(input[:, 2, :])

        input_combined = torch.cat((word_emb, user_bot_emb, cur_emb), 2)

        rnn_out, self.hidden = self.rnn(input_combined, self.hidden)
        output = self.linear(self.hidden).view(-1, 3)

        if calc_softmax:
            probs = self.softmax(output)
            return self.hidden, probs
        else:
            return self.hidden, output

    def save(self, path='data/models/sentence/model.pytorch'):
        torch.save(self, path)
        return True

    @staticmethod
    def load(path='data/models/sentence/model.pytorch'):
        return torch.load(path)
