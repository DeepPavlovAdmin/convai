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
    parser = argparse.ArgumentParser("Retrieve response according to log-likelihood from separate response set")

    parser.add_argument("model_prefix",
            help="Path to the model prefix (i.e. without _model.npz or _state.pkl postfix)")

    parser.add_argument("context",
            help="File of input contexts (tokenized text with one example per line)")

    parser.add_argument("responses",
            help="File of potential responses; the responses for each context must be tab separated tokenized text, and correspond to the context of the same line in the context file")

    parser.add_argument("output",
            help="Output text file containing the retrieved responses (tokenized responses with each line being the retrived response to the corresponding context line)")

    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")

    return parser.parse_args()

def main():
    args = parse_args()
    state = prototype_state()

    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    # For simplicity, we force the batch size to be one
    state['bs'] = 1
    model = DialogEncoderDecoder(state) 

    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    

    eval_batch = model.build_eval_function()

    contexts = [[]]
    lines = open(args.context, "r").readlines()
    if len(lines):
        contexts = [x.strip() for x in lines]


    potential_responses_set = open(args.responses, "r").readlines()
    most_probable_responses_string = ''
    
    print('Retrieval started...')

    for context_idx, context in enumerate(contexts):
        if context_idx % 100 == 0:
            print '     processing example: ' + str(context_idx) + ' / ' + str(len(contexts))
        potential_responses = potential_responses_set[context_idx].strip().split('\t')

        most_probable_response_loglikelihood = -1.0
        most_probable_response = ''

        for potential_response_idx, potential_response in enumerate(potential_responses):
            # Convert contexts into list of ids
            dialogue = []
            if len(context) == 0:
                dialogue = [model.eos_sym]
            else:
                sentence_ids = model.words_to_indices(context.split())
                # Add eos tokens
                if len(sentence_ids) > 0:
                    if not sentence_ids[0] == model.eos_sym:
                        sentence_ids = [model.eos_sym] + sentence_ids
                    if not sentence_ids[-1] == model.eos_sym:
                        sentence_ids += [model.eos_sym]
                else:
                    sentence_ids = [model.eos_sym]

                dialogue += sentence_ids

            response = model.words_to_indices(potential_response.split())
            if len(response) > 0:
                if response[0] == model.eos_sym:
                    del response[0]
                if not response[-1] == model.eos_sym:
                    response += [model.eos_sym]

            dialogue += response
            dialogue = numpy.asarray(dialogue, dtype='int32').reshape((len(dialogue), 1))

            dialogue_reversed = model.reverse_utterances(dialogue)

            dialogue_mask = numpy.ones((len(dialogue), 1), dtype='float32') 
            dialogue_weight = numpy.ones((len(dialogue), 1), dtype='float32')
            dialogue_reset_mask = numpy.zeros((1), dtype='float32')
            dialogue_ran_vectors = model.rng.normal(size=(1,model.latent_gaussian_per_utterance_dim)).astype('float32')
            dialogue_ran_vectors = numpy.tile(dialogue_ran_vectors, (len(dialogue),1,1))

            dialogue_drop_mask = numpy.ones((len(dialogue), 1), dtype='float32')

            c, _, _, _, _ = eval_batch(dialogue, dialogue_reversed, len(dialogue), dialogue_mask, dialogue_weight, dialogue_reset_mask, dialogue_ran_vectors, dialogue_drop_mask)
            c = c / len(dialogue)

            print 'c', c

            if (potential_response_idx == 0) or (-c > most_probable_response_loglikelihood):
                most_probable_response_loglikelihood = -c
                most_probable_response = potential_response

        most_probable_responses_string += most_probable_response + '\n'

    print('Retrieval finished.')
    print('Saving to file...')
     
    # Write to output file
    output_handle = open(args.output, "w")
    output_handle.write(most_probable_responses_string)
    output_handle.close()

    print('Saving to file finished.')
    print('All done!')

if __name__ == "__main__":
    # THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python retrieve.py tests/models/1450723451.38_testmodel tests/data/tvalid_contexts.txt tests/data/tvalid_potential_responses.txt Out.txt --verbose

    # awk -F "\t" '{print $1}' GeneratedResponses.txt &> First_GeneratedResponses.txt
    
    main()

