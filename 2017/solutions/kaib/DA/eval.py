#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

import pdb

# Parameters
# ==================================================

# Data Parameters
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
##tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

#tf.flags.DEFINE_string("positive_data_file", "./data/chatbot/squad_filterlengthy.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/chatbot/movie_unique_filterlengthy.txt", "Data source for the negative data.")

#tf.flags.DEFINE_string("positive_data_file", "./data/qa.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/cc.txt", "Data source for the negative data.")


#tf.flags.DEFINE_string("positive_data_file", "data/convai_processed_qa.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "data/convai_processed_cc.txt", "Data source for the negative data.")

tf.flags.DEFINE_string("positive_data_file", "data/test.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "data/test1000_cc.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test, match = data_helpers.load_data_and_labels_ev(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["What is the daily student paper at Notre Dame called", "What is everything to someone and nothing to everyone else"]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print(np.size(x_test))
m_a = []
# Build vocabulary
for m_data in match:
    if np.size(x_test[0]) < len(m_data):
        m_temp = m_data[0:np.size(x_test[0])]
    else:
        m_zero = np.zeros((np.size(x_test[0]) -  len(m_data)), dtype=np.float)
        m_temp = np.concatenate([m_data, m_zero], 0)
    m_a.append(m_temp)
m = np.array(m_a)



print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_m = graph.get_operation_by_name("input_m").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        #x_batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        #m_batches = data_helpers.batch_iter3(list(m), FLAGS.batch_size, 1, shuffle=False)
        #x_batches, m_batches = data_helpers.batch_iter2(list(x_test), list(m), FLAGS.batch_size, 1, shuffle=False)
        #m_batches = data_helpers.batch_iter2(list(m), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch, m_test_batch in data_helpers.batch_iter2(list(x_test), list(m), FLAGS.batch_size, 1, shuffle=False):
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, input_m: m_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
