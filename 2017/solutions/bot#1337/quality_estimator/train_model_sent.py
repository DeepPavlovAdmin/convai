import pickle
import torch
import torch.utils.data
import torch.nn as nn
from sys import argv
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from collections import Counter
from models import UtteranceModel


def load_dialogs_and_labels(filename):
    with open(filename, 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # X_train = X_train[:100]
    # X_test = X_test[:100]
    # y_train = y_train[:100]
    # y_test = y_test[:100]

    X_train = torch.LongTensor(X_train)
    y_train = torch.LongTensor(y_train)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    X_test = torch.LongTensor(X_test)
    y_test = torch.LongTensor(y_test)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    return train_loader, test_loader


def load_sent_labels(filename):
    with open(filename, 'rb') as f:
        labels = pickle.load(f)

    return labels


def forward_pass_sent(model, data):
    model.zero_grad()
    model.hidden = Variable(torch.zeros(50, 1, 128))
    hidden, out = model(data, True)
    return out


def measure_model_quality(model, loss_function, test_loader, prev_best_f1=0, with_save=True):
    avg_loss = 0
    y_pred = []
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        data, target = Variable(data), Variable(target)
        out = forward_pass_sent(model, data)
        loss = loss_function(out, target)
        avg_loss += loss.data[0]

        top_n, top_i = out.data.topk(1)
        y_pred += top_i.resize_(top_i.size()[0]).tolist()

    print("Test loss: {}".format(avg_loss / len(test_loader.dataset)))

    y_test = test_loader.dataset.target_tensor.tolist()

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Test F1: {}".format(f1))

    print(classification_report(y_test, y_pred))

    if f1 >= prev_best_f1 and with_save:
        prev_best_f1 = f1
        model.save()

    return prev_best_f1


# NOTE: For some reason it is not working with cuda: RuntimeError: Expected hidden size (1, 16, 128), got (50, 16, 128)
def main():
    train_loader, test_loader = load_dialogs_and_labels('data/sent_data.pickle')

    model = UtteranceModel()
    loss_function = nn.NLLLoss()
    # model.cuda()
    # loss_function.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    prev_best_f1 = 0
    for epoch in range(10):
        avg_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = Variable(data), Variable(target)
            out = forward_pass_sent(model, data)
            loss = loss_function(out, target)
            avg_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        print("Loss: {}".format(avg_loss / len(train_loader.dataset)))

        prev_best_f1 = measure_model_quality(model, loss_function, test_loader, prev_best_f1)


def main_test():
    train_loader, test_loader = load_dialogs_and_labels('data/sent_data.pickle')
    model = UtteranceModel.load()
    loss_function = nn.NLLLoss()

    measure_model_quality(model, loss_function, test_loader, 0, False)


if __name__ == '__main__':
    if argv[1] == 'test':
        main_test()
    else:
        main()
