import tensorflow as tf
import numpy as np
import cPickle as pkl
import argparse
import copy
import json
import sys
import os

from estimators import Estimator, SHORT_TERM_MODE, LONG_TERM_MODE

import features as _features
import inspect

ALL_FEATURES = []
for name, obj in inspect.getmembers(_features):
    if inspect.isclass(obj) and name not in ['SentimentIntensityAnalyzer', 'Feature']:
        ALL_FEATURES.append(name)


MODE_TO_TARGET = {
    'short_term': 'r',
    'long_term': 'R'
}

MODE_TO_FLAG = {
    'short_term': SHORT_TERM_MODE,
    'long_term': LONG_TERM_MODE
}


TARGET_TO_FEATURES = {
    'r': [
        'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_Article',
        'Similarity_CandidateUser',
        'NonStopWordOverlap', 'BigramOverlap', 'TrigramOverlap', 'EntityOverlap',
        'GenericTurns',
        'WhWords', 'IntensifierWords', 'ConfusionWords', 'ProfanityWords', 'Negation',
        'LastUserLength', 'CandidateLength',
        'DialogActCandidate', 'DialogActLastUser',
        'SentimentScoreCandidate', 'SentimentScoreLastUser'
    ],
    'R': [
        'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_LastK',
        'AverageWordEmbedding_kUser', 'AverageWordEmbedding_Article',
        'Similarity_CandidateUser',
        'Similarity_CandidateLastK', 'Similarity_CandidateLastK_noStop',
        'Similarity_CandidateKUser', 'Similarity_CandidateKUser_noStop',
        'Similarity_CandidateArticle', 'Similarity_CandidateArticle_noStop',
        'NonStopWordOverlap', 'BigramOverlap', 'TrigramOverlap', 'EntityOverlap',
        'GenericTurns',
        'WhWords', 'IntensifierWords', 'ConfusionWords', 'ProfanityWords', 'Negation',
        'DialogLength', 'LastUserLength', 'CandidateLength', 'ArticleLength',
        'DialogActCandidate', 'DialogActLastUser',
        'SentimentScoreCandidate', 'SentimentScoreLastUser'
    ]
}


def get_data(files, target, feature_list=None, val_prop=0.1, test_prop=0.1):
    """
    Load data to train ranker. Build `k` fold cross validation train/val data
    :param files: list of data files to load
    :param target: field of each dictionary instance to estimate
        either 'R' for final dialog score, or 'r' for immediate reward
    :param feature_list: list of feature names (str) to load.
        if none will take the ones defined in TARGET_TO_FEATURES.
    :param val_prop: proportion of data to consider for validation set.
        Will also define the number of train/valid folds
    :param test_prop: proportion of data to consider for test set
    :return: collections of train x & y, valid x & y, and one test x & y
        x = numpy array of size (data, feature_length)
          ie: np.array( [[f1, f2, ..., fn], ..., [f1, f2, ..., fn]] )
        y = array of label values to predict
    """
    assert target in ['R', 'r'], "Unknown target: %s" % target

    print "Loading data..."
    raw_data = {}  # map file name to list of dictionaries
    ''' FORMAT:
    {
       file_name : [ {-}, {-}, ..., {-} ],
       ...
    }
    with {-} being the dictionary object containing context, article, etc...
    '''
    n = 0  # total number of examples
    for data_file in files:
        if target == 'r' and 'data/voted_data_' in data_file:
            with open(data_file, 'rb') as handle:
                raw_data[data_file] = json.load(handle)
                n += len(raw_data[data_file])
                # get the time id of the data
                # file_ids.append(data_file.split('_')[-1].replace('pkl', ''))
        elif target == 'R' and 'data/full_data_' in data_file:
            with open(data_file, 'rb') as handle:
                raw_data[data_file] = json.load(handle)
                n += len(raw_data[data_file])
                # get the time id of the data
                # file_ids.append(data_file.split('_')[-1].replace('pkl', ''))
        else:
            print "Warning: will not consider file %s because target=%s" % (data_file, target)
    print "got %d examples" % n

    # Build map from article to filename to data_idx to avoid having overlap between train/valid/test sets
    article2file2id = {}
    ''' FORMAT: {
        article : { file_name: [idx, idx, ..., idx],
                    ...
                  },
        ...
    } '''
    for data_file, data in raw_data.iteritems():
        for idx, msg in enumerate(data):
            article = msg['article']
            if article not in article2file2id:
                article2file2id[article] = {}

            if data_file not in article2file2id[article]:
                article2file2id[article][data_file] = [idx]
            else:
                article2file2id[article][data_file].append(idx)
    print "got %d unique articles" % len(article2file2id)

    train_max_n = int(n * (1-val_prop-test_prop))  # size of training data
    valid_max_n = int(n * val_prop)  # size of valid data
    test_max_n  = int(n * test_prop) # size of test data

    test_data, remain_data = {}, {}  # store map from filename to list of indices
    test_n, remain_n = 0, 0          # number of examples
    for article, file2id in article2file2id.iteritems():
        # add to test set
        if test_n < test_max_n:
            for data_file, indices in file2id.iteritems():
                if data_file not in test_data:
                    test_data[data_file] = []
                test_data[data_file].extend(indices)
                test_n += len(indices)
        # keep the remaining for train & valid k-fold
        else:
            for data_file, indices in file2id.iteritems():
                if data_file not in remain_data:
                    remain_data[data_file] = []
                remain_data[data_file].extend(indices)
                remain_n += len(indices)

    # create list of Feature instances
    if feature_list is None:
        feature_list = TARGET_TO_FEATURES[target]
    feature_objects = _features.get(article=None, context=None, candidate=None, feature_list=feature_list)
    input_size = np.sum([f.dim for f in feature_objects])
    del feature_objects  # now that we have the input_size, don't need those anymore

    # construct data to save & return
    remain_x = []  # np.zeros((remain_n, input_size))
    remain_y = []
    test_x = []    # np.zeros((test_n, input_size))
    test_y = []

    print "building data..."
    for x, y, data in [(remain_x, remain_y, remain_data), (test_x, test_y, test_data)]:
        for data_file, indices in data.iteritems():
            # load the required features for that file
            features = {}
            feature_path = data_file.replace('.json', '.features')
            for feat in feature_list:
                with open("%s/%s.json" % (feature_path, feat), 'rb') as handle:
                    features[feat] = json.load(handle)
            for idx in indices:
                msg = raw_data[data_file][idx]
                # create input features for this msg:
                tmp = np.concatenate( [features[feat][idx] for feat in feature_list] )
                x.append(tmp.tolist())
                # set y labels
                if target == 'r':
                    if int(msg[target]) == -1: y.append(0)
                    elif int(msg[target]) == 1: y.append(1)
                    else: print "ERROR: unknown immediate reward value: %s" % msg[target]
                else:
                    y.append(msg[target])

    remain_x = np.array(remain_x)
    assert remain_x.shape == (remain_n, input_size), "%s != %s" % (remain_x.shape, (remain_n, input_size))
    remain_y = np.array(remain_y)
    assert len(remain_y) == remain_n, "%d != %d" % (len(remain_y), remain_n)
    test_x = np.array(test_x)
    assert test_x.shape == (test_n, input_size), "%s != %s" % (test_x.shape, (test_n, input_size))
    test_y = np.array(test_y)
    assert len(test_y) == test_n, "%d != %d" % (len(test_y), test_n)

    # reformat train & valid to build k-folds:
    trains = []
    valids = []
    n = len(remain_y)
    for k_fold in range(n / valid_max_n):
        start = valid_max_n * k_fold  # start index of validation set
        stop =  valid_max_n * (k_fold+1)  # stop index of validation set
        # define validation set for this fold
        tmp_valid_x = remain_x[start: stop]
        tmp_valid_y = remain_y[start: stop]
        valids.append((tmp_valid_x, tmp_valid_y))
        print "[fold %d] valid: %s" % (k_fold+1, tmp_valid_x.shape)
        # define training set for this fold
        # print "[fold %d] train indices: %s" % (k+fold+1, np.array(range(stop, n+start)) % n)
        tmp_train_x = remain_x[np.array(range(stop, n+start)) % n]
        tmp_train_y = remain_y[np.array(range(stop, n+start)) % n]
        trains.append((tmp_train_x, tmp_train_y))
        print "[fold %d] train: %s" % (k_fold+1, tmp_train_x.shape)
    print "test: %s" % (test_x.shape,)

    return trains, valids, (test_x, test_y), feature_list


def sample_parameters(t):
    """
    randomly choose a set of parameters t times
    """
    # force 3*300 features for input of size > 900
    default_features = ['AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_Article']
    features = filter(lambda f : f not in default_features, ALL_FEATURES)
    
    # map from hidden sizes to "used_before" flag
    hidd_sizes = dict(
        [(k, False) for k in [
            (900, 300), (700, 300), (700, 100), (500, 50), (500, 100), (300, 50),
            (900, 500, 200), (900, 300, 50), (700, 500, 50), (700, 100, 50),
            (500, 700, 300), (500, 400, 100), (500, 100, 50), (300, 500, 100),
            (900, 500, 500, 100), (900, 600, 300, 100), (700, 300, 300, 100), (700, 500, 300, 100), (500, 200, 200, 50),
            (800, 500, 300, 100, 50), (900, 600, 300, 100, 50), (800, 300, 300, 300, 50)
        ]]
    )
    activations = ['swish', 'relu', 'sigmoid']
    optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    batch_sizes = [32, 64, 128, 256, 512, 1024]

    feats, hidds, activs, optims, lrs, drs, bss = [], [], [], [], [], [], []
    # sample parameters
    for _ in range(t):
        sampled_features = copy.deepcopy(default_features)
        n = np.random.randint(len(features)/2, len(features)+1)  # number of features to sample: between half and all
        sampled_features.extend(np.random.choice(features, n, replace=False))

        new_sizes = [h for h in hidd_sizes.keys() if not hidd_sizes[h]]
        if len(new_sizes) == 0:
            # reset all `used_before` flags to False
            for h in hidd_sizes.keys(): hidd_sizes[h] = False
            new_sizes = [h for h in hidd_sizes.keys() if not hidd_sizes[h]]
            assert len(new_sizes) == len(hidd_sizes)
        idx = np.random.choice(len(new_sizes))
        sampled_hidd = new_sizes[idx]
        hidd_sizes[sampled_hidd] = True  # flag this size to be sampled

        feats.append(sampled_features)
        hidds.append(sampled_hidd)
        activs.append(np.random.choice(activations))
        optims.append(np.random.choice(optimizers))
        lrs.append(np.random.choice(learning_rates))
        drs.append(np.random.choice(dropout_rates))
        bss.append(np.random.choice(batch_sizes))

    return feats, hidds, activs, optims, lrs, drs, bss


def load_previous_model(prefix):
    """
    :param prefix: example: models/short_term/0.643257/1510158946.66_VoteEstimator_
    """
    print "Loading previous model arguments..."
    with open("%sargs.pkl" % prefix, 'rb') as handle:
        model_args = pkl.load(handle)

    if len(model_args) == 12:
        data, \
        hidden_dims, hidden_dims_extra, activation, \
        optimizer, learning_rate, \
        model_path, model_id, model_name, \
        batch_size, dropout_rate, pretrained = model_args
    elif len(model_args) == 8:
        data, \
        hidden_dims, activation, \
        optimizer, learning_rate, \
        model_id, \
        batch_size, dropout_rate = model_args
        # reconstruct missing parameters
        hidden_dims_extra = [hidden_dims[-1]]
        model_name = prefix.split(model_id)[1].replace('_', '')
        pretrained = None
    else:
        print "WARNING: %d model arguments." % len(model_args)
        print "data + %s" % (model_args[1:],)
        return

    # reconstruct model_path just in case it has been moved:
    model_path = prefix.split(model_id)[0]
    if model_path.endswith('/'):
        model_path = model_path[:-1]  # ignore the last '/'

    # Load previously saved model timings
    with open("%stimings.pkl" % prefix, 'rb') as handle:
        train_accuracies, valid_accuracies = pkl.load(handle)

    n_folds = len(train_accuracies)
    max_train = [max(train_accuracies[i]) for i in range(n_folds)]
    max_valid = [max(valid_accuracies[i]) for i in range(n_folds)]
    print "prev. max train accuracies: %s" % (max_train,)
    print "prev. max valid accuracies: %s" % (max_valid,)
    train_acc = np.mean(max_train)
    valid_acc = np.mean(max_valid)
    print "prev. best avg. train accuracy: %g" % train_acc
    print "prev. best avg. valid accuracy: %g" % valid_acc

    return data, \
           hidden_dims, hidden_dims_extra, activation, \
           optimizer, learning_rate, \
           model_path, model_id, model_name, \
           batch_size, dropout_rate, pretrained, \
           train_accuracies, valid_accuracies


def main(args):
    if args.explore:
        # sample a bunch of parameters, and run those experiments
        feats, hidds, activs, optims, lrs, drs, bss = sample_parameters(args.explore)
        best_args = []  # store the best combination
        best_valid_acc = -100000.  # store the best validation accuracy
        best_model = None  # store the best model id
        valid_threshold = args.threshold  # accuracy must be higher for model to be saved
        print "Will try %d different configurations..." % args.explore
        for idx in range(args.explore):
            with tf.Session() as sess:
                try:
                    # print sampled parameters
                    print "\n[%d] sampled features:\n%s" % (idx+1, feats[idx])
                    print "[%d] sampled hidden_sizes: %s" % (idx+1, hidds[idx])
                    print "[%d] extra hidden_sizes: %s" % (idx+1, hidds[idx][-1])
                    print "[%d] sampled activation: %s" % (idx+1, activs[idx])
                    print "[%d] sampled optimizer: %s" % (idx+1, optims[idx])
                    print "[%d] sampled learning rate: %g" % (idx+1, lrs[idx])
                    print "[%d] sampled dropout rate: %g" % (idx+1, drs[idx])
                    print "[%d] sampled batch size: %d" % (idx+1, bss[idx])

                    # Load datasets
                    data = get_data(args.data, MODE_TO_TARGET[args.mode], feature_list=feats[idx])
                    n_folds = len(data[0])
                    print "[%d] Building the network..." % (idx+1,)
                    estimator = Estimator(
                        data,
                        hidds[idx],
                        [hidds[idx][-1]],
                        activs[idx],
                        optims[idx],
                        lrs[idx],
                        model_path='models/%s' % args.mode
                    )
                    print "[%d] Training the network..." % (idx+1,)
                    estimator.train(
                        sess,
                        MODE_TO_FLAG[args.mode],
                        args.patience,
                        bss[idx],
                        drs[idx],
                        save=False,  # don't save for now
                        pretrained=None,
                        verbose=False
                    )
                    # Only consider last `n_folds` accuracies!
                    max_train = [max(estimator.train_accuracies[i]) for i in range(-n_folds, 0)]
                    max_valid = [max(estimator.valid_accuracies[i]) for i in range(-n_folds, 0)]
                    print "[%d] max train accuracies: %s" % (idx+1, max_train)
                    print "[%d] max valid accuracies: %s" % (idx+1, max_valid)
                    train_acc = np.mean(max_train)
                    valid_acc = np.mean(max_valid)
                    print "[%d] best avg. train accuracy: %g" % (idx+1, train_acc)
                    print "[%d] best avg. valid accuracy: %g" % (idx+1, valid_acc)

                    # save now if we got a good model
                    if valid_acc > valid_threshold:
                        estimator.save(sess)

                    # update variables if we got better model
                    if valid_acc > best_valid_acc:
                        print "[%d] got better accuracy! new: %g > old: %g" % (idx+1, valid_acc, best_valid_acc)
                        best_valid_acc = valid_acc
                        best_model = estimator.model_id
                        best_args = [feats[idx], hidds[idx], activs[idx], optims[idx], lrs[idx], drs[idx], bss[idx]]
                    else:
                        print "[%d] best validation accuracy is still %g" % (idx+1, best_valid_acc)

                # end of try block, catch CTRL+C errors to print current results
                except KeyboardInterrupt as e:
                    print e
                    print "best model: %s" % best_model
                    print "with parameters:"
                    print " - features:\n%s"     % (best_args[0],)
                    print " - hidden_sizes: %s"  % (best_args[1],)
                    print " - activation: %s"    % (best_args[2],)
                    print " - optimizer: %s"     % (best_args[3],)
                    print " - learning rate: %g" % (best_args[4],)
                    print " - dropout rate: %g"  % (best_args[5],)
                    print " - batch size: %d"    % (best_args[6],)
                    print "with average valid accuracy: %g" % best_valid_acc
                    sys.exit()

            # end of tensorflow session, reset for the next graph
            tf.reset_default_graph()

        # end of exploration, print best results:
        print "done!"
        print "best model: %s" % best_model
        print "with parameters:"
        print " - features:\n%s"     % (best_args[0],)
        print " - hidden_sizes: %s"  % (best_args[1],)
        print " - activation: %s"    % (best_args[2],)
        print " - optimizer: %s"     % (best_args[3],)
        print " - learning rate: %g" % (best_args[4],)
        print " - dropout rate: %g"  % (best_args[5],)
        print " - batch size: %d"    % (best_args[6],)
        print "with average valid accuracy: %g" % best_valid_acc

    else:
        # run one experiment with provided parameters
        # load previously trained model.
        if args.previous_model:
            old_data, \
            hidden_sizes, hidden_sizes_extra, activation, \
            optimizer, learning_rate, \
            model_path, model_id, model_name, \
            batch_size, dropout_rate, pretrained, \
            train_accuracies, valid_accuracies = load_previous_model(args.previous_model)
            # Load provided dataset, but with the same features as previous model
            data = get_data(args.data, MODE_TO_TARGET[args.mode], feature_list=old_data[-1])
            # set pretrained to this model name
            pretrained = (model_path, model_id, model_name)
            # now update this model name to not override previous one
            model_name = "%s" % args.previous_model.replace('models', '').replace('.', '').replace('//', '').replace('/', '.')
            # store previous_accuracies
            previous_accuracies = (train_accuracies, valid_accuracies)
        else:
            # else, build current parameters
            data = get_data(args.data, MODE_TO_TARGET[args.mode])
            hidden_sizes = args.hidden_sizes
            hidden_sizes_extra = [args.hidden_sizes[-1]]
            activation = args.activation
            optimizer = args.optimizer
            learning_rate = args.learning_rate
            model_id = None  # keep the default one
            model_name = 'Estimator'
            batch_size = args.batch_size
            dropout_rate = args.dropout_rate
            pretrained = None
            previous_accuracies = None

        n_folds = len(data[0])
        print "Building the network..."
        estimator = Estimator(
            data,
            hidden_sizes,
            hidden_sizes_extra,
            activation,
            optimizer,
            learning_rate,
            model_path='models/%s' % args.mode, model_id=model_id, model_name=model_name
        )

        with tf.Session() as sess:
            print "Training the network..."
            estimator.train(
                sess,
                MODE_TO_FLAG[args.mode],
                args.patience,
                batch_size,
                dropout_rate,
                save=True,
                pretrained=pretrained,
                previous_accuracies=previous_accuracies,
                verbose=True
            )
            max_train = [max(estimator.train_accuracies[i]) for i in range(-n_folds, 0)]
            max_valid = [max(estimator.valid_accuracies[i]) for i in range(-n_folds, 0)]
            print "max train accuracies: %s" % (max_train,)
            print "max valid accuracies: %s" % (max_valid,)
            train_acc = np.mean(max_train)
            valid_acc = np.mean(max_valid)
            print "best avg. train accuracy: %g" % train_acc
            print "best avg. valid accuracy: %g" % valid_acc
        print "done."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs='+', type=str, help="List of files to consider for training")
    parser.add_argument("mode", choices=['short_term', 'long_term'], type=str, help="What reward should the estimator predict")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    parser.add_argument("-ex", "--explore", type=int, default=None, help="Number of times to sample parameters. If None, will use the one provided")
    parser.add_argument("-t",  "--threshold", type=float, default=0.63, help="minimum accuracy to reach in order to save the model (only used in exploration mode)")
    # training parameters:
    parser.add_argument("-pm", "--previous_model", default=None, help="path and prefix_ of the model to continue training from")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during training")
    parser.add_argument("-p",  "--patience", type=int, default=20, help="Number of training steps to wait before stoping when validatiaon accuracy doesn't increase")
    # network architecture:
    parser.add_argument("-hs", "--hidden_sizes", nargs='+', type=int, default=[500, 300, 100, 10], help="List of hidden sizes for the network")
    parser.add_argument("-ac", "--activation", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function")
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.1, help="Probability of dropout layer")
    parser.add_argument("-op", "--optimizer", choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], default='adam', help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    args = parser.parse_args()
    print "\n%s\n" % args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    main(args)

