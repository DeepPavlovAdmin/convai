#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs
import search
import utils

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state

logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def parse_args():
    parser = argparse.ArgumentParser("Computes Recall@1, Recall@2 and Recall@5 for model ranked responses given ground truth responses")

    parser.add_argument("true_responses",
            help="File containing ground truth responses; one response per line (tokenized text)")

    parser.add_argument("ranked_responses",
            help="File containing ranked responses according to model; responses for each example must be tab separated (tokenized text) and each line must correspond to the line of the true_responses")

    return parser.parse_args()

def main():
    args = parse_args()

    true_responses = [[]]
    lines = open(args.true_responses, "r").readlines()
    if len(lines):
        true_responses = [x.strip() for x in lines]

    lines = open(args.ranked_responses, "r").readlines()
    if len(lines):
        ranked_responses = [x.strip() for x in lines]

    assert len(true_responses) == len(ranked_responses)

    recall_at_one = 0
    recall_at_two = 0
    recall_at_five = 0

    print 'Computing recall metrics...'
    for response_idx, true_response in enumerate(true_responses):
        retrieval_index = -1
        for resp_idx, resp in enumerate(ranked_responses[response_idx].split('\t')):
            if resp == true_response:
                retrieval_index = resp_idx
                break

        if retrieval_index < 0:
            print 'Error for response index: ', response_idx

        if retrieval_index == 0:
            recall_at_one += 1
            recall_at_two += 1
            recall_at_five += 1
        elif retrieval_index == 1:
            recall_at_two += 1
            recall_at_five += 1
        elif retrieval_index <= 4:
            recall_at_five += 1


    recall_at_one = (float(recall_at_one) * 100.0) / float(len(true_responses))
    recall_at_two = (float(recall_at_two) * 100.0) / float(len(true_responses))
    recall_at_five = (float(recall_at_five) * 100.0) / float(len(true_responses))
    
    print '     Recall@1: ', recall_at_one
    print '     Recall@2: ', recall_at_two
    print '     Recall@5: ', recall_at_five
        

if __name__ == "__main__":
    main()

