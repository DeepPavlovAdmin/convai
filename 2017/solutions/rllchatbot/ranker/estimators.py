import tensorflow as tf
import time
import cPickle as pkl


ACTIVATIONS = {
    'swish': lambda x: x * tf.sigmoid(x),
    'relu': tf.nn.relu,
    'sigmoid': tf.sigmoid
}


OPTIMIZERS = {
    'sgd': tf.train.GradientDescentOptimizer,
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer
}


SHORT_TERM_MODE = 0
LONG_TERM_MODE = 1


class Estimator(object):
    def __init__(self, data, hidden_dims, hidden_dims_extra, activation, optimizer, learning_rate, model_path='models', model_id=None, model_name='Estimator'):
        """
        Build the estimator for either short term or long term reward based on mode
        :param data: train, valid, test data to use
        :param hidden_dims: list of ints specifying the size of each hidden layer for the first estimator
        :param hidden_dims_extr: list of ints specifying the size of each hidden layer for the second estimator
        :param activation: tensor activation function to use at each layer
        :param optimizer: tensorflow optimizer object to train the network
        :param learning_rate: learning rate for the optimizer
        :param model_path: path of folders where to save the model
        :param model_id: if None, set to creation time
        :param model_name: name for saved model files
        """
        self.trains, self.valids, (self.x_test, self.y_test), self.feature_list = data
        self.n_folds = len(self.trains)
        _, self.input_dim = self.trains[0][0].shape

        self.hidden_dims = hidden_dims
        self.hidden_dims_extra = hidden_dims_extra
        self.activation = activation
        self.optimizer = optimizer
        self.lr = learning_rate

        self.model_path = model_path
        if model_id: self.model_id = model_id
        else: self.model_id = str(time.time())
        if model_name: self.model_name = model_name
        else: self.model_name = 'Estimator'

        self.SHORT_TERM = tf.constant(SHORT_TERM_MODE, dtype=tf.int8, name="SHORT_TERM_MODE")
        self.LONG_TERM  = tf.constant(LONG_TERM_MODE, dtype=tf.int8, name="LONG_TERM_MODE")

        self.build()

    def build(self):
        """
        Build the actual neural net, define the predictions, loss, train operator, and accuracy
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="input_layer")  # (bs, feat_size)
        self.y = tf.placeholder(tf.int64, shape=[None, ], name="labels")  # (bs,1)

        # scalar to decide in which mode we are: SHORT_TERM or LONG_TERM
        self.mode = tf.placeholder(tf.int8, name="estimator_type")

        # Fully connected dense layers
        h_fc = self.x  # (bs, in)
        for idx, hidd in enumerate(self.hidden_dims):
            h_fc = tf.layers.dense(inputs=h_fc,
                                   units=hidd,
                                   # kernel_initializer = Initializer function for the weight matrix.
                                   # bias_initializer: Initializer function for the bias.
                                   activation=ACTIVATIONS[self.activation],
                                   name='dense_layer%d' % (idx + 1))  # (bs, hidd)

        # define extra layers for long term estimator:
        def _extra_layers():
            h_fc_extra = h_fc
            for idx, hidd_extra in enumerate(self.hidden_dims_extra):
                h_fc_extra = tf.layers.dense(inputs=h_fc_extra,
                                             units=hidd_extra,
                                             # kernel_initializer = Initializer function for the weight matrix.
                                             # bias_initializer: Initializer function for the bias.
                                             activation=ACTIVATIONS[self.activation],
                                             name='extra_dense_layer_%d' % (idx + 1))  # (bs, hidd)
            return h_fc_extra
        # add extra layers to current network if LONG TERM estimator
        h_fc = tf.cond(tf.equal(self.mode, self.LONG_TERM),
                       true_fn  = _extra_layers,
                       false_fn = lambda: h_fc)

        # Dropout layer
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # proba of keeping the neuron
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob, name='dropout_layer')

        # Output layer: linear activation to 1 scalar if LONG TERM estimator, to 2 digits if SHORT TERM
        logits = tf.cond(tf.equal(self.mode, self.LONG_TERM),
                         true_fn  = lambda: tf.layers.dense(inputs=h_fc_drop, units=1),
                         false_fn = lambda: tf.layers.dense(inputs=h_fc_drop, units=2),
                         name='logits_layer')  # shape (bs, 1) or (bs, 2)

        # Prediction tensor: logit's first axis for LONG_TERM, logit's argmax for SHORT_TERM
        self.predictions = tf.cond(tf.equal(self.mode, self.LONG_TERM),
                                   true_fn  = lambda: logits[:, 0],
                                   # convert argmax to float32 to be compatible with LONG_TERM logit's type:
                                   false_fn = lambda: tf.cast(tf.argmax(logits, axis=1), tf.float32),
                                   name='prediction_tensor')  # shape (bs,)

        # Confidence tensor: only used in the SHORT_TERM mode: probability of up-vote, in LONG_TERM mode same as predictions
        self.confidences = tf.cond(tf.equal(self.mode, self.LONG_TERM),
                                   true_fn  = lambda: logits[:, 0],
                                   false_fn = lambda: tf.nn.softmax(logits, dim=1)[:, 1],
                                   name='confidence_tensor')  # shape (bs,)

        # define loss tensor for SHORT TERM estimator:
        def _short_term_loss():
            # create one-hot labels
            onehot_labels = tf.one_hot(indices=tf.cast(self.y, tf.int32), depth=2)  # from shape (bs,) to (bs, 2)
            # define the cross-entropy loss
            return tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        # define loss tensor for LONG TERM estimator:
        def _long_term_loss():
            # labels = tf.expand_dims(self.y, axis=1)  # from shape (bs,) to (bs, 1)
            return tf.losses.mean_squared_error(self.y, logits[:, 0])
        # Loss tensor: mean squared error for LONG TERM, softmax cross entropy for SHORT_TERM
        self.loss = tf.cond(tf.equal(self.mode, self.LONG_TERM),
                            true_fn  = _long_term_loss,
                            false_fn = _short_term_loss,
                            name = "loss_tensor")

        # Train operator:
        optimizer = OPTIMIZERS[self.optimizer](learning_rate=self.lr)
        self.train_step = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # define accuracy for short term estimator:
        def _short_term_accuracy():
            # define class label (0,1) and class probabilities:
            # classes = tf.argmax(logits, axis=1, name="shortterm_pred_classes")   # (bs,)
            classes = tf.cast(self.predictions, tf.int64)  # (bs,)
            # Accuracy tensor:
            correct_predictions = tf.equal(classes, self.y)
            return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        # Accuracy tensor:
        self.accuracy = tf.cond(tf.equal(self.mode, self.LONG_TERM),
                                true_fn  = lambda: - self.loss,
                                false_fn = _short_term_accuracy,
                                name = "accuracy_tensor")

        # Once graph is built, create a saver for the model:
        # Add an op to initialize the variables.
        self.init_op = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def train(self, session, mode, patience, batch_size, dropout_rate, save=True, pretrained=None, previous_accuracies=None, verbose=True):
        """
        :param session: tensorflow session
        :param mode: Estimator.SHORT_TERM or Estimator.LONG_TERM
        :param patience: number of times to continue training when no improvement on validation
        :param batch_size: number of examples per batch
        :param dropout_rate: probability of drop out
        :param save: decide if we save the model & its parameters
        :param pretrained: list of (model_path, model_id, model_name) for the pretrained model, or None
        :param previous_accuracies: list of (train_accuracies, valid_accuracies) from pretrained model, or None
        :param verbose: print statements all over the place or not
        """
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.pretrained = pretrained
        if previous_accuracies:
            self.train_accuracies = previous_accuracies[0]
            self.valid_accuracies = previous_accuracies[1]
        else:
            self.train_accuracies = []
            self.valid_accuracies = []

        # Perform k-fold cross validation: train/valid on k different part of the data
        fold = 0
        for (x_train, y_train), (x_valid, y_valid) in zip(self.trains, self.valids):
            fold += 1
            train_accs = []  # accuracies for this fold, to be added at the end of the fold
            valid_accs = []  # accuracies for this fold, to be added at the end of the fold
            n, _ = x_train.shape

            best_valid_acc = -100000.
            p = patience

            print "begin fold %d" % fold,
            # reset model to its pretrained state
            if pretrained and len(pretrained) == 3:
                self.load(session, pretrained[0], pretrained[1], pretrained[2])
            # initialize model variables
            else:
                print "reset model to initial state"
                session.run(self.init_op)

            for epoch in range(20000):  # will probably stop before 20k epochs due to early stop
                # do 1 epoch: go through all training_batches
                for idx in range(0, n, batch_size):
                    _, loss = session.run(
                        [self.train_step, self.loss],
                        feed_dict={self.x: x_train[idx: idx + batch_size],
                                   self.y: y_train[idx: idx + batch_size],
                                   self.keep_prob: 1.0 - dropout_rate,
                                   self.mode: mode}
                    )
                    step = idx / batch_size
                    # if step % 10 == 0:
                    #     print "[fold %d] epoch %d - step %d - training loss: %g" % (fold, epoch+1, step, loss)
                if verbose: print "[fold %d] epoch %d - step %d - training loss: %g" % (fold, epoch+1, step, loss)
                # print "------------------------------"
                # Evaluate (so no dropout) on training set
                train_acc = self.evaluate(mode, x_train, y_train)
                if verbose: print "[fold %d] epoch %d: train accuracy: %g" % (fold, epoch+1, train_acc)
                train_accs.append(train_acc)

                # Evaluate (so no dropout) on validation set
                valid_acc = self.evaluate(mode, x_valid, y_valid)
                if verbose: print "[fold %d] epoch %d: valid accuracy: %g" % (fold, epoch+1, valid_acc)
                valid_accs.append(valid_acc)

                # early stop
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc  # set best acc
                    p = patience  # reset patience to initial value bcs score improved
                    if save:
                        # save model when improved in that fold
                        self.save(session, save_args=False, save_timings=False)
                else:
                    p -= 1
                if verbose: print "[fold %d] epoch %d: patience: %d" % (fold, epoch+1, p)
                if p == 0:
                    break

            self.train_accuracies.append(train_accs)
            self.valid_accuracies.append(valid_accs)

            if save:
                # save the arguments and the timings when done this fold
                self.save(session, save_model=False)
            if verbose: print "------------------------------"

    def test(self, mode, x=None, y=None):
        """
        evaluate model on test set
        :param mode: SHORT_TERM or LONG_TERM: different accuracy definition
        specify `x` and `y` if we want to evaluate on different set than self.test
        """
        if x is None or y is None:
            x, y = self.x_test, self.y_test
        test_acc = self.evaluate(mode, x, y)
        return test_acc

    def evaluate(self, mode, x, y):
        """
        :param mode: SHORT_TERM or LONG_TERM: different accuracy definition
        """
        # Evaluate accuracy, so no dropout
        acc = self.accuracy.eval(
            feed_dict={self.x: x, self.y: y, self.keep_prob: 1.0, self.mode: mode}
        )
        return acc

    def predict(self, mode, x):
        """
        :param mode: SHORT_TERM or LONG_TERM: different prediction definition
        :return: prediction tensor to the user and the model's confidence:
          if mode is LONG_TERM, then confidence and predictions are the same
          if mode is SHORT_TERM, confidence is the probability of an up-vote
        """
        preds = self.predictions.eval(
            feed_dict={self.x: x, self.keep_prob: 1.0, self.mode: mode}
        )
        confs = self.confidences.eval(
            feed_dict={self.x: x, self.keep_prob: 1.0, self.mode: mode}
        )
        return preds, confs

    def save(self, session, save_model=True, save_args=True, save_timings=True):
        prefix = "%s/%s_%s" % (self.model_path, self.model_id, self.model_name)
        # save the tensorflow graph variables
        if save_model:
            saved_path = self.saver.save(session, "%s_model.ckpt" % prefix)
            print "Model saved in file: %s" % saved_path
        # save the arguments used to create that object
        if save_args:
            data = [
                self.trains,
                self.valids,
                (self.x_test, self.y_test),
                self.feature_list
            ]
            with open("%s_args.pkl" % prefix, 'wb') as handle:
                pkl.dump(
                    [data, self.hidden_dims, self.hidden_dims_extra, self.activation,
                     self.optimizer, self.lr,
                     self.model_path, self.model_id, self.model_name,
                     self.batch_size, self.dropout_rate, self.pretrained],
                    handle,
                    pkl.HIGHEST_PROTOCOL
                )
            print "Args (and data) saved."
        # Save timings measured during training
        if save_timings:
            with open("%s_timings.pkl" % prefix, 'wb') as handle:
                pkl.dump(
                    [self.train_accuracies, self.valid_accuracies],
                    handle,
                    pkl.HIGHEST_PROTOCOL
                )
            print "Timings saved."

    def load(self, session, model_path=None, model_id=None, model_name=None):
        """
        Load graph from given model path, id, name. Default is this model path, id, name.
        """
        if model_path is None or model_id is None or model_name is None:
            model_path = self.model_path
            model_id   = self.model_id
            model_name = self.model_name
        self.saver.restore(session, "%s/%s_%s_model.ckpt" % (model_path, model_id, model_name))
        print "Model restored to %s/%s_%s" % (model_path, model_id, model_name)


