#!/usr/bin/env python

import argparse
import cPickle
import numpy
import logging
import time

import os
import search

from dialog_encdec import DialogEncoderDecoder
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
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    parser.add_argument(
        "model_prefix",
        help="Path to the model prefix (without _model.npz or _state.pkl)"
    )
    parser.add_argument(
        "--ignore-unk",
        action="store_false",  # default is True: generation of unknown words is disabled
        help="Disables the generation of unknown words (<unk> tokens)"
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",  # default is False: using random search
        help="Use beam search instead of random search"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",  # default is False
        help="Be verbose"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    state = prototype_state()

    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"
    timing_path = args.model_prefix + "_timing.npz"

    with open(state_path, 'r') as src:
        state.update(cPickle.load(src))
    with open(timing_path, 'r') as src:
        timings = dict(numpy.load(src))

    state['compute_training_updates'] = False

    logging.basicConfig(
        level=getattr(logging, state['level']),
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s"
    )

    print "\nLoaded previous state, model, timings:"
    print "state:"
    print state
    print "timings:"
    print timings

    print "\nBuilding model..."
    model = DialogEncoderDecoder(state)

    sampler = search.RandomSampler(model)
    if args.beam_search:
        sampler = search.BeamSampler(model)

    if os.path.isfile(model_path):
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    print "build.\n"

    context = []
    while True:
        line = raw_input("user: ")
        context.append( "<first_speaker> <at> "+line+" </s> ")
        print "context: ", [' '.join(context[-4:])]
        context_samples, context_costs = sampler.sample([' '.join(context[-4:])], ignore_unk=args.ignore_unk, verbose=args.verbose, return_words=True)

        print "bot:", context_samples
        context.append(context_samples[0][0]+" </s> ")
        print "cost:", context_costs

if __name__ == "__main__":
    main()

