#!/usr/bin/env python
"""
This script computes dialogue embeddings for dialogues found in a text file.
"""

#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys
import math

import os
import numpy
import codecs
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
    parser = argparse.ArgumentParser("Compute dialogue embeddings from model")

    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("dialogues",
            help="File of input dialogues (tab separated)")

    parser.add_argument("output",
            help="Output file")
    
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")

    parser.add_argument("--use-second-last-state",
            action="store_true", default=False,
            help="Outputs the second last dialogue encoder state instead of the last one")

    return parser.parse_args()

def compute_encodings(joined_contexts, model, model_compute_encoding, output_second_last_state = False):
    # TODO Fix seqlen below
    seqlen = 600
    context = numpy.zeros((seqlen, len(joined_contexts)), dtype='int32')
    context_lengths = numpy.zeros(len(joined_contexts), dtype='int32')
    second_last_utterance_position = numpy.zeros(len(joined_contexts), dtype='int32')


    for idx in range(len(joined_contexts)):
        context_lengths[idx] = len(joined_contexts[idx])
        if context_lengths[idx] < seqlen:
            context[:context_lengths[idx], idx] = joined_contexts[idx]
        else:
            # If context is longer tham max context, truncate it and force the end-of-utterance token at the end
            context[:seqlen, idx] = joined_contexts[idx][0:seqlen]
            context[seqlen-1, idx] = model.eos_sym
            context_lengths[idx] = seqlen

        eos_indices = list(numpy.where(context[:context_lengths[idx], idx] == model.eos_sym)[0])

        if len(eos_indices) > 1:
            second_last_utterance_position[idx] = eos_indices[-2]
        else:
            second_last_utterance_position[idx] = context_lengths[idx]

    n_samples = len(joined_contexts)

    # Generate the reversed context
    reversed_context = model.reverse_utterances(context)

    encoder_states = model_compute_encoding(context, reversed_context, seqlen+1)
    hidden_states = encoder_states[-2] # hidden state for the "context" encoder, h_s,
                                       # and last hidden state of the utterance "encoder", h
    #hidden_states = encoder_states[-1] # mean for the stochastic latent variable, z

    if output_second_last_state:
        second_last_hidden_state = numpy.zeros((hidden_states.shape[1], hidden_states.shape[2]), dtype='float64')
        for i in range(hidden_states.shape[1]):
            second_last_hidden_state[i, :] = hidden_states[second_last_utterance_position[i], i, :]

        return second_last_hidden_state
    else:
        return hidden_states[-1, :, :]


def main(model_prefix, dialogue_file, use_second_last_state):
    state = prototype_state()

    state_path = model_prefix + "_state.pkl"
    model_path = model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    state['bs'] = 10

    model = DialogEncoderDecoder(state) 
    
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")

    contexts = [[]]
    lines = open(dialogue_file, "r").readlines()
    if len(lines):
        contexts = [x.strip() for x in lines]

    model_compute_encoding = model.build_encoder_function()
    dialogue_encodings = []

    # Start loop
    joined_contexts = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(contexts)) / float(model.bs)))
    for context_id, context_sentences in enumerate(contexts):
        # Convert contexts into list of ids
        joined_context = []

        if len(context_sentences) == 0:
            joined_context = [model.eos_sym]
        else:
            joined_context = model.words_to_indices(context_sentences.split())

            if joined_context[0] != model.eos_sym:
                joined_context = [model.eos_sym] + joined_context

            if joined_context[-1] != model.eos_sym:
                joined_context += [model.eos_sym]

        joined_contexts.append(joined_context)

        if len(joined_contexts) == model.bs:
            batch_index = batch_index + 1
            logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_index, batch_total))
            encs = compute_encodings(joined_contexts, model, model_compute_encoding, use_second_last_state)
            for i in range(len(encs)):
                dialogue_encodings.append(encs[i])

            joined_contexts = []


    if len(joined_contexts) > 0:
        logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_total, batch_total))
        encs = compute_encodings(joined_contexts, model, model_compute_encoding, use_second_last_state)
        for i in range(len(encs)):
            dialogue_encodings.append(encs[i])

    return dialogue_encodings

if __name__ == "__main__":
    args = parse_args()

    # Compute encodings
    dialogue_encodings = main(args.model_prefix, args.dialogues, args.use_second_last_state)

    # Save encodings to disc
    cPickle.dump(dialogue_encodings, open(args.output + '.pkl', 'w'))


    #  THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python compute_dialogue_embeddings.py tests/models/1462302387.69_testmodel tests/data/tvalid_contexts.txt Latent_Variable_Means --verbose --use-second-last-state
