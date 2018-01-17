import tensorflow as tf
import numpy as np
import cPickle as pkl
import argparse
import pyprind
import copy
import json
import time
import sys
import os

from estimators import Estimator, ACTIVATIONS, OPTIMIZERS, SHORT_TERM_MODE, LONG_TERM_MODE


def load_previous_model(prefix):
    """
    """
    print "Loading model %s ..." % prefix
    # Load previously saved model arguments
    with open("%sargs.pkl" % prefix, 'rb') as handle:
        data, \
        hidden_dims, hidden_dims_extra, activation, \
        optimizer, learning_rate, \
        model_path, model_id, model_name, \
        batch_size, dropout_rate, pretrained = pkl.load(handle)

    # reconstruct model_path just in case it has been moved:
    model_path = prefix.split(model_id)[0]
    if model_path.endswith('/'):
        model_path = model_path[:-1]  # ignore the last '/'

    # Load previously saved model timings
    with open("%stimings.pkl" % prefix, 'rb') as handle:
        train_accuracies, valid_accuracies = pkl.load(handle)

    max_train = [max(accs) for accs in train_accuracies]
    max_valid = [max(accs) for accs in valid_accuracies]
    print "prev. max train accuracies: %s" % (max_train,)
    print "prev. max valid accuracies: %s" % (max_valid,)
    # consider only the last `n_folds` accuracies! Accuracies before that may be from another objective
    n_folds = len(data[0])
    max_train = max_train[-n_folds:]
    max_valid = max_valid[-n_folds:]
    train_acc = np.mean(max_train)
    valid_acc = np.mean(max_valid)
    print "prev. best avg. train accuracy: %g" % train_acc
    print "prev. best avg. valid accuracy: %g" % valid_acc

    return data, \
        hidden_dims, hidden_dims_extra, activation, \
        optimizer, learning_rate, \
        model_path, model_id, model_name, \
        batch_size, dropout_rate, pretrained


def build_graph(data, hidden_dims, hidden_dims_extra, activation, optimizer, learning_rate, model_path, model_id, model_name):
    model_graph = tf.Graph()
    with model_graph.as_default():
        estimator = Estimator(
            data,
            hidden_dims, hidden_dims_extra, activation,
            optimizer, learning_rate,
            model_path, model_id, model_name
        )
    return model_graph, estimator


def main(args):
    data_s, \
        hidden_dims_s, hidden_dims_extra_s, activation_s, \
        optimizer_s, learning_rate_s, \
        model_path_s, model_id_s, model_name_s, \
        batch_size_s, dropout_rate_s, pretrained_s = load_previous_model(args.short_term_model)

    data_l, \
        hidden_dims_l, hidden_dims_extra_l, activation_l, \
        optimizer_l, learning_rate_l, \
        model_path_l, model_id_l, model_name_l, \
        batch_size_l, dropout_rate_l, pretrained_l = load_previous_model(args.long_term_model)


    n_folds = len(data_s[0])
    assert n_folds == len(data_l[0])

    print "Building the networks..."
    graph_s, estim_s = build_graph(data_s, hidden_dims_s, hidden_dims_extra_s, activation_s, optimizer_s, learning_rate_s, model_path_s, model_id_s, model_name_s)
    graph_l, estim_l = build_graph(data_l, hidden_dims_l, hidden_dims_extra_l, activation_l, optimizer_l, learning_rate_l, model_path_l, model_id_l, model_name_l)

    sess_s = tf.Session(graph=graph_s)
    sess_l = tf.Session(graph=graph_l)

    with sess_s.as_default():
        with graph_s.as_default():
            print "Reset short term network parameters..."
            estim_s.load(sess_s, model_path_s, model_id_s, model_name_s)

    with sess_l.as_default():
        with graph_l.as_default():
            print "Reset long term network parameters..."
            estim_l.load(sess_l, model_path_l, model_id_l, model_name_l)

    with sess_s.as_default():
        with graph_s.as_default():
            print "Testing the short term network..."
            test_acc = estim_s.test(SHORT_TERM_MODE)
            print "test accuracy: %g" % test_acc

    with sess_l.as_default():
        with graph_l.as_default():
            print "Testing the long term network..."
            test_acc = estim_l.test(LONG_TERM_MODE)
            print "test accuracy OTHER: %g" % test_acc

    '''Code below is to train again one of the model'''
    # print "\nContinue training the network..."
    # estimator.train(
    #     sess,
    #     MODE_TO_FLAGS[args.mode],
    #     args.patience,
    #     batch_size,
    #     dropout_rate,
    #     save=False,
    #     pretrained=(model_path, model_id, model_name),
    #     previous_accuracies=(train_accuracies, valid_accuracies),
    #     verbose=True
    # )
    # Consider the newly added accuracies!
    # max_train = [max(estimator.train_accuracies[i]) for i in range(-n_folds, 0)]
    # max_valid = [max(estimator.valid_accuracies[i]) for i in range(-n_folds, 0)]
    # print "max train accuracies: %s" % (max_train,)
    # print "max valid accuracies: %s" % (max_valid,)
    # train_acc = np.mean(max_train)
    # valid_acc = np.mean(max_valid)
    # print "best avg. train accuracy: %g" % train_acc
    # print "best avg. valid accuracy: %g" % valid_acc
    # print "Re-testing the network..."
    # test_acc = estimator.test(MODE_TO_FLAGS[args.mode])
    # print "test accuracy: %g" % test_acc

    with sess_s.as_default():
        with graph_s.as_default():
            print "Get train, valid, test prediction..."
            trains, valids, (x_test, y_test), feature_list = data_s
            train_acc, valid_acc = [], []
            for fold in range(len(trains)):
                preds, confs = estim_s.predict(SHORT_TERM_MODE, trains[fold][0])
                same = float(np.sum(preds == trains[fold][1]))
                # TODO: compute true/false positives/negatives
                acc  = same/len(preds)
                train_acc.append(acc)
                print "[fold %d] train acc: %g/%d=%g" % (fold+1, same, len(preds), acc)

                preds, confs = estim_s.predict(SHORT_TERM_MODE, valids[fold][0])
                same = float(np.sum(preds == valids[fold][1]))
                # TODO: compute true/false positives/negatives
                acc  = same/len(preds)
                valid_acc.append(acc)
                print "[fold %d] valid acc: %g/%d=%g" % (fold+1, same, len(preds), acc)

            print "avg. train acc. %g" % np.mean(train_acc)
            print "avg. valid acc. %g" % np.mean(valid_acc)
            preds, confs = estim_s.predict(SHORT_TERM_MODE, x_test)
            same = float(np.sum(preds == y_test))
            # TODO: compute true/false positives/negatives
            acc = same/len(preds)
            print "test acc: %g/%d=%g" % (same, len(preds), acc)


    with sess_l.as_default():
        with graph_l.as_default():
            print "Get train, valid, test prediction..."
            trains, valids, (x_test, y_test), feature_list = data_l
            train_acc, valid_acc = [], []
            for fold in range(len(trains)):
                preds, confs = estim_l.predict(LONG_TERM_MODE, trains[fold][0])
                same = -np.sum((preds - trains[fold][1])**2)
                # TODO: plot x:labels y:predictions ~ confusion matrix/plot
                acc  = same/len(preds)
                train_acc.append(acc)
                print "[fold %d] train acc: %g/%d=%g" % (fold+1, same, len(preds), acc)

                preds, confs = estim_l.predict(LONG_TERM_MODE, valids[fold][0])
                same = -np.sum((preds - valids[fold][1])**2)
                # TODO: plot x:labels y:predictions ~ confusion matrix/plot
                acc  = same/len(preds)
                valid_acc.append(acc)
                print "[fold %d] valid acc: %g/%d=%g" % (fold+1, same, len(preds), acc)

            print "avg. train acc. %g" % np.mean(train_acc)
            print "avg. valid acc. %g" % np.mean(valid_acc)
            preds, confs = estim_l.predict(LONG_TERM_MODE, x_test)
            same = -np.sum((preds - y_test)**2)
            # TODO: plot x:labels y:predictions ~ confusion matrix/plot
            acc = same/len(preds)
            print "test acc: %g/%d=%g" % (same, len(preds), acc)

    print "done."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("short_term_model", type=str, help="Short term estimator prefix to load")
    parser.add_argument("long_term_model", type=str, help="Long term estimator prefix to load")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    args = parser.parse_args()
    print "\n%s\n" % args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    main(args)

