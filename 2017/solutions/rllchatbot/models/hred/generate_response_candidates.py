#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import numpy.random
import codecs
import search
import utils

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state


from scipy.sparse import lil_matrix



logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def parse_args():
    parser = argparse.ArgumentParser("Given a set of training contexts, training responses and test contexts, returns a set of potential test responses.")

    parser.add_argument("dictionary",
            help="File of dictionary")

    parser.add_argument("training_contexts",
            help="File of training contexts (tokenized text with one example per line)")

    parser.add_argument("training_responses",
            help="File of potential responses (tokenied text with each response corresponding to the context on the same line in the training_contexts file)")

    parser.add_argument("test_contexts",
            help="File of test contexts (tokenized text with one example per line)")

    parser.add_argument("test_responses",
            help="File of test responses (tokenized text with one example per line corresponding to the context on the same line in the test_contexts file)")

    parser.add_argument("potential_responses", type=int,
            help="Number of potential responses to retrieve")

    parser.add_argument("retrieval_method",
            help="Choose between methods 'random', 'random-truth' and 'tf-idf'")

    parser.add_argument("output",
            help="Output file with potential responses; the potential responses for each test context are tab separated tokenized text, and correspond to the context of the same line in the test context file")

    return parser.parse_args()


def words_to_indices(seq, str_to_idx, unk_sym=0):
    """
    Converts a list of words to a list
    of word ids. Use unk_sym if a word is not
    known.
    """
    return [str_to_idx.get(word, unk_sym) for word in seq]
    

def main():
    args = parse_args()

    raw_dict = cPickle.load(open(args.dictionary, 'r'))
    str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict])
    idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq, _ in raw_dict])
    document_freq = dict([(tok_id, df) for _, tok_id, _, df in raw_dict])
    idim = len(str_to_idx)

    # Fix the random number generator
    rng = numpy.random.RandomState(1234)

    training_contexts = [[]]
    lines = open(args.training_contexts, "r").readlines()
    if len(lines):
        training_contexts = [x.strip() for x in lines]

    training_contexts_len = len(training_contexts)

    # Populate matrix for training_contexts
    if (args.retrieval_method.lower() == 'tf-idf'):
        print('Populating matrix for training contexts...')
        training_contexts_matrix = lil_matrix((len(training_contexts), idim), dtype=numpy.float32)
        for training_context_idx, training_context in enumerate(training_contexts):
            if training_context_idx % 100 == 0:
                print '     processing training context: ' + str(training_context_idx) + ' / ' + str(len(training_contexts))

            word_indices = words_to_indices(training_context.split(), str_to_idx)
            for word_index in word_indices:
                training_contexts_matrix[training_context_idx, word_index] += numpy.log(1 + training_contexts_len/max(1.0, document_freq[word_index]))

            # Normalize by L2 norm
            training_contexts_matrix[training_context_idx, :] = training_contexts_matrix[training_context_idx, :] / numpy.linalg.norm(training_contexts_matrix[training_context_idx, :].toarray())


    training_responses = [[]]
    lines = open(args.training_responses, "r").readlines()
    if len(lines):
        training_responses = [x.strip() for x in lines]

    test_contexts = [[]]
    lines = open(args.test_contexts, "r").readlines()
    if len(lines):
        test_contexts = [x.strip() for x in lines]

    test_responses = [[]]
    lines = open(args.test_responses, "r").readlines()
    if len(lines):
        test_responses = [x.strip() for x in lines]

    assert len(training_contexts) == len(training_responses)
    assert len(test_contexts) == len(test_responses)
    assert (args.potential_responses > 0) and (args.potential_responses <= len(training_responses))

    if args.retrieval_method.lower() == 'random-truth':
        # If random-truth is selected with a single potential response to be retrieved,
        # that potential response will be the only ground truth test, which is not acceptable...
        assert args.potential_responses > 1 
    
    print('Retrieval started...')
    retrieved_potential_test_responses = ''
    for test_context_idx, test_context in enumerate(test_contexts):
        potential_responses = []
        if test_context_idx % 100 == 0:
            print '     processing test context: ' + str(test_context_idx) + ' / ' + str(len(test_contexts))

        if (args.retrieval_method.lower() == 'random'):
            all_indices = range(len(training_responses))
            rng.shuffle(all_indices)
            indices_to_retrieve = all_indices[0:args.potential_responses]
            for index_to_retrieve in indices_to_retrieve:
                potential_responses += [training_responses[index_to_retrieve]]

        elif (args.retrieval_method.lower() == 'random-truth'):
            potential_responses += [test_responses[test_context_idx]]

            all_indices = range(len(training_responses))
            rng.shuffle(all_indices)
            indices_to_retrieve = all_indices[1:args.potential_responses]
            for index_to_retrieve in indices_to_retrieve:
                potential_responses += [training_responses[index_to_retrieve]]

        elif (args.retrieval_method.lower() == 'tf-idf'):
            test_context_vector = numpy.zeros((idim), dtype='float32')
            test_context_word_indices = words_to_indices(test_context.split(), str_to_idx)
            for word_index in test_context_word_indices:
                test_context_vector[word_index] += numpy.log(1 + training_contexts_len/max(1.0, document_freq[word_index]))

            cs = training_contexts_matrix.dot(test_context_vector)

            cs_largest_indices = numpy.argpartition(cs, -args.potential_responses)[-args.potential_responses:]
            for index_to_retrieve in cs_largest_indices:
                potential_responses += [training_responses[index_to_retrieve]]
        else:
            print 'ERROR! Please choose between the following retrieval methods: random, random-truth, tf-idf'


        retrieved_potential_test_responses += '\t'.join(potential_responses) + '\n'

    print('Retrieval finished.')
    print('Saving to file...')
     
    # Write to output file
    output_handle = open(args.output, "w")
    output_handle.write(retrieved_potential_test_responses)
    output_handle.close()

    print('Saving to file finished.')
    print('All done!')


if __name__ == "__main__":
    main()

    # python generate_response_candidates.py tests/models/1450723451.38_testmodel tests/data/tvalid_contexts.txt tests/data/tvalid_responses.txt tests/data/tvalid_contexts.txt tests/data/tvalid_responses.txt 2 tf-idf Out.txt
    

