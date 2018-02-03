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
import math

from dialog_encdec import DialogEncoderDecoder 
from numpy_compat import argpartition
from state import * 
from data_iterator import get_test_iterator

import matplotlib
matplotlib.use('Agg')
import pylab

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    
    parser.add_argument("test_dialogues",
            type=str, help="File of test data (one dialogue per line)")

    return parser.parse_args()

def main():
    args = parse_args()
    state = prototype_state()
   
    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src)) 
    
    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    state['lr'] = 0.01
    state['bs'] = 2

    state['compute_training_updates'] = False
    state['apply_meanfield_inference'] = True

    model = DialogEncoderDecoder(state)
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    


    mf_update_batch = model.build_mf_update_function()
    mf_reset_batch = model.build_mf_reset_function()

    saliency_batch = model.build_saliency_eval_function()

    test_dialogues = open(args.test_dialogues, 'r').readlines()
    for test_dialogue_idx, test_dialogue in enumerate(test_dialogues):
        #print 'Visualizing dialogue: ', test_dialogue
        test_dialogue_split = test_dialogue.split()
        # Convert dialogue into list of ids
        dialogue = []
        if len(test_dialogue) == 0:
            dialogue = [model.eos_sym]
        else:
            sentence_ids = model.words_to_indices(test_dialogue_split)
            # Add eos tokens
            if len(sentence_ids) > 0:
                if not sentence_ids[0] == model.eos_sym:
                    sentence_ids = [model.eos_sym] + sentence_ids
                if not sentence_ids[-1] == model.eos_sym:
                    sentence_ids += [model.eos_sym]
            else:
                sentence_ids = [model.eos_sym]


            dialogue += sentence_ids

        if len(dialogue) > 3:
            if ((dialogue[-1] == model.eos_sym)
             and (dialogue[-2] == model.eod_sym)
             and (dialogue[-3] == model.eos_sym)):
                del dialogue[-1]
                del dialogue[-1]

        
        dialogue = numpy.asarray(dialogue, dtype='int32').reshape((len(dialogue), 1))
        #print 'dialogue', dialogue
        dialogue_reversed = model.reverse_utterances(dialogue)

        max_batch_sequence_length = len(dialogue)
        bs = state['bs']

        # Initialize batch with zeros
        batch_dialogues = numpy.zeros((max_batch_sequence_length, bs), dtype='int32')
        batch_dialogues_reversed = numpy.zeros((max_batch_sequence_length, bs), dtype='int32')
        batch_dialogues_mask = numpy.zeros((max_batch_sequence_length, bs), dtype='float32')
        batch_dialogues_reset_mask = numpy.zeros((bs), dtype='float32')
        batch_dialogues_drop_mask = numpy.ones((max_batch_sequence_length, bs), dtype='float32')

        # Fill in batch with values
        batch_dialogues[:,0]  = dialogue[:, 0]
        batch_dialogues_reversed[:,0] = dialogue_reversed[:, 0]
        #batch_dialogues  = dialogue
        #batch_dialogues_reversed = dialogue_reversed


        batch_dialogues_ran_gaussian_vectors = numpy.zeros((max_batch_sequence_length, bs, model.latent_gaussian_per_utterance_dim), dtype='float32')
        batch_dialogues_ran_uniform_vectors = numpy.zeros((max_batch_sequence_length, bs, model.latent_piecewise_per_utterance_dim), dtype='float32')

        eos_sym_list = numpy.where(batch_dialogues[:, 0] == state['eos_sym'])[0]
        if len(eos_sym_list) > 1:
            second_last_eos_sym = eos_sym_list[-2]
        else:
            print 'WARNING: dialogue does not have at least two EOS tokens!'

        batch_dialogues_mask[:, 0] = 1.0
        batch_dialogues_mask[0:second_last_eos_sym+1, 0] = 0.0

        print '###'
        print '###'
        print '###'
        if False==True:
            mf_reset_batch()
            for i in range(10):
                print  '  SGD Update', i

                batch_dialogues_ran_gaussian_vectors[second_last_eos_sym:, 0, :] = model.rng.normal(loc=0, scale=1, size=model.latent_gaussian_per_utterance_dim)
                batch_dialogues_ran_uniform_vectors[second_last_eos_sym:, 0, :] = model.rng.uniform(low=0.0, high=1.0, size=model.latent_piecewise_per_utterance_dim)


                training_cost, kl_divergence_cost_acc, kl_divergences_between_piecewise_prior_and_posterior, kl_divergences_between_gaussian_prior_and_posterior = mf_update_batch(batch_dialogues, batch_dialogues_reversed, max_batch_sequence_length, batch_dialogues_mask, batch_dialogues_reset_mask, batch_dialogues_ran_gaussian_vectors, batch_dialogues_ran_uniform_vectors, batch_dialogues_drop_mask)

                print '     training_cost', training_cost
                print '     kl_divergence_cost_acc', kl_divergence_cost_acc
                print '     kl_divergences_between_gaussian_prior_and_posterior',  numpy.sum(kl_divergences_between_gaussian_prior_and_posterior)
                print '     kl_divergences_between_piecewise_prior_and_posterior', numpy.sum(kl_divergences_between_piecewise_prior_and_posterior)


        batch_dialogues_ran_gaussian_vectors[second_last_eos_sym:, 0, :] = model.rng.normal(loc=0, scale=1, size=model.latent_gaussian_per_utterance_dim)
        batch_dialogues_ran_uniform_vectors[second_last_eos_sym:, 0, :] = model.rng.uniform(low=0.0, high=1.0, size=model.latent_piecewise_per_utterance_dim)

        gaussian_saliency, piecewise_saliency = saliency_batch(batch_dialogues, batch_dialogues_reversed, max_batch_sequence_length, batch_dialogues_mask, batch_dialogues_reset_mask, batch_dialogues_ran_gaussian_vectors, batch_dialogues_ran_uniform_vectors, batch_dialogues_drop_mask)

        if test_dialogue_idx < 2:
            print 'gaussian_saliency', gaussian_saliency.shape, gaussian_saliency
            print 'piecewise_saliency', piecewise_saliency.shape, piecewise_saliency

        gaussian_sum = 0.0
        piecewise_sum = 0.0
        for i in range(second_last_eos_sym+1, max_batch_sequence_length):
            gaussian_sum += gaussian_saliency[dialogue[i, 0]]
            piecewise_sum += piecewise_saliency[dialogue[i, 0]]

        gaussian_sum = max(gaussian_sum, 0.0000000000001)
        piecewise_sum = max(piecewise_sum, 0.0000000000001)


        print '###'
        print '###'
        print '###'

        print 'Topic: ', ' '.join(test_dialogue_split[0:second_last_eos_sym])

        print ''
        print 'Response', ' '.join(test_dialogue_split[second_last_eos_sym+1:max_batch_sequence_length])
        gaussian_str = ''
        piecewise_str = ''
        for i in range(second_last_eos_sym+1, max_batch_sequence_length):
            gaussian_str += str(gaussian_saliency[dialogue[i, 0]]/gaussian_sum) + ' '
            piecewise_str += str(piecewise_saliency[dialogue[i, 0]]/piecewise_sum) + ' '




        print 'Gaussian_saliency', gaussian_str
        print 'Piecewise_saliency', piecewise_str

        #print ''
        #print 'HEY', gaussian_saliency[:, 0].argsort()[-3:][::-1]
        print ''
        print 'Gaussian Top 3 Words: ', model.indices_to_words(list(gaussian_saliency[:].argsort()[-3:][::-1]))
        print 'Piecewise Top 3 Words: ', model.indices_to_words(list(piecewise_saliency[:].argsort()[-3:][::-1]))

        print 'Gaussian Top 5 Words: ', model.indices_to_words(list(gaussian_saliency[:].argsort()[-5:][::-1]))
        print 'Piecewise Top 5 Words: ', model.indices_to_words(list(piecewise_saliency[:].argsort()[-5:][::-1]))

        print 'Gaussian Top 7 Words: ', model.indices_to_words(list(gaussian_saliency[:].argsort()[-7:][::-1]))
        print 'Piecewise Top 7 Words: ', model.indices_to_words(list(piecewise_saliency[:].argsort()[-7:][::-1]))


    logger.debug("All done, exiting...")

if __name__ == "__main__":
    main()
