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
    parser = argparse.ArgumentParser("Ranks responses according to log-likelihood from separate response set")

    parser.add_argument("model_prefix",
            help="Path to the model prefix (i.e. without _model.npz or _state.pkl postfix)")

    parser.add_argument("context",
            help="File of input contexts (tokenized text with one example per line)")

    parser.add_argument("responses",
            help="File of potential responses; the responses for each context must be tab separated tokenized text, and correspond to the context of the same line in the context file")

    parser.add_argument("output",
            help="Output text file containing the tab responses in ranked order (tokenized responses with each line being the retrived response to the corresponding context line). The first response is the response ranked highest, second response is the response ranked second highest and so on.")

    parser.add_argument("--mf_inference_steps", type=int, default=0, help="Mean field inference steps. Zero means no mean field inference is carried out.")

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

    state['bs'] = 20
    state['compute_training_updates'] = False

    if args.mf_inference_steps > 0:
        state['lr'] = 0.01
        state['apply_meanfield_inference'] = True

    model = DialogEncoderDecoder(state) 

    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    

    eval_batch = model.build_eval_function()

    if args.mf_inference_steps > 0:
        mf_update_batch = model.build_mf_update_function()
        mf_reset_batch = model.build_mf_reset_function()
        
    contexts = [[]]
    lines = open(args.context, "r").readlines()
    if len(lines):
        contexts = [x.strip() for x in lines]

    lines = open(args.responses, "r").readlines()
    if len(lines):
        potential_responses_set = [x.strip() for x in lines]
    
    print('Building data batches...')

    # This is the maximum sequence length we can process on a 12 GB GPU with large HRED models
    max_sequence_length = 80*(80/state['bs'])

    assert len(potential_responses_set) == len(contexts)

    # Note we assume that each example has the same number of potential responses
    # The code can be adapted, however, by simply counting the total number of responses to consider here...
    examples_to_process = len(contexts) * len(potential_responses_set[0].strip().split('\t'))

    all_dialogues = numpy.zeros((max_sequence_length, examples_to_process), dtype='int32')
    all_dialogues_reversed = numpy.zeros((max_sequence_length, examples_to_process), dtype='int32')
    all_dialogues_mask = numpy.zeros((max_sequence_length, examples_to_process), dtype='float32')
    all_dialogues_reset_mask = numpy.zeros((examples_to_process), dtype='float32')
    all_dialogues_drop_mask = numpy.ones((max_sequence_length, examples_to_process), dtype='float32')

    all_dialogues_len = numpy.zeros((examples_to_process), dtype='int32')

    example_idx = -1

    for context_idx, context in enumerate(contexts):
        if context_idx % 100 == 0:
            print '     processing context idx: ' + str(context_idx) + ' / ' + str(len(contexts))

        potential_responses = potential_responses_set[context_idx].strip().split('\t')

        most_probable_response_loglikelihood = -1.0
        most_probable_response = ''

        for potential_response_idx, potential_response in enumerate(potential_responses):
            example_idx += 1

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

            if len(response) > 3:
                if ((response[-1] == model.eos_sym)
                 and (response[-2] == model.eod_sym)
                 and (response[-3] == model.eos_sym)):
                    del response[-1]
                    del response[-1]

            dialogue += response

            if context_idx == 0:
                print 'DEBUG INFO'
                print 'dialogue', dialogue

            if len(numpy.where(numpy.asarray(dialogue) == state['eos_sym'])[0]) < 3:
                print 'POTENTIAL PROBLEM WITH DIALOGUE CONTEXT IDX', context_idx
                print '     dialogue', dialogue


            # Trim beginning of dialogue words if the dialogue is too long... this should be a rare event.
            if len(dialogue) > max_sequence_length:
                if state['do_generate_first_utterance']:
                    dialogue = dialogue[len(dialogue)-max_sequence_length:len(dialogue)]
                else: # CreateDebate specific setting
                    dialogue = dialogue[0:max_sequence_length-1]
                    if not dialogue[-1] == state['eos_sym']:
                        dialogue += [state['eos_sym']]

                    


            dialogue = numpy.asarray(dialogue, dtype='int32').reshape((len(dialogue), 1))

            dialogue_reversed = model.reverse_utterances(dialogue)
            dialogue_mask = numpy.ones((len(dialogue)), dtype='float32') 
            dialogue_weight = numpy.ones((len(dialogue)), dtype='float32')
            dialogue_drop_mask = numpy.ones((len(dialogue)), dtype='float32')

            # Add example to large numpy arrays...
            all_dialogues[0:len(dialogue), example_idx] = dialogue[:,0]
            all_dialogues_reversed[0:len(dialogue), example_idx] = dialogue_reversed[:,0]
            all_dialogues_mask[0:len(dialogue), example_idx] = dialogue_mask
            all_dialogues_drop_mask[0:len(dialogue), example_idx] = dialogue_drop_mask

            all_dialogues_len[example_idx] = len(dialogue)


    sorted_dialogue_indices = numpy.argsort(all_dialogues_len)

    all_dialogues_cost = numpy.ones((examples_to_process), dtype='float32')

    batch_count = int(numpy.ceil(float(examples_to_process) / float(state['bs'])))

    print('Computing costs on batches...')
    for batch_idx in reversed(range(batch_count)):
        if batch_idx % 10 == 0:
            print '     processing batch idx: ' + str(batch_idx) + ' / ' + str(batch_count)

        example_index_start = batch_idx * state['bs']
        example_index_end = min((batch_idx+1) * state['bs'], examples_to_process)
        example_indices = sorted_dialogue_indices[example_index_start:example_index_end]

        current_batch_size = len(example_indices)

        max_batch_sequence_length = all_dialogues_len[example_indices[-1]]

        # Initialize batch with zeros
        batch_dialogues = numpy.zeros((max_batch_sequence_length, state['bs']), dtype='int32')
        batch_dialogues_reversed = numpy.zeros((max_batch_sequence_length, state['bs']), dtype='int32')
        batch_dialogues_mask = numpy.zeros((max_batch_sequence_length, state['bs']), dtype='float32')
        batch_dialogues_weight = numpy.ones((max_batch_sequence_length, state['bs']), dtype='float32')
        batch_dialogues_reset_mask = numpy.zeros((state['bs']), dtype='float32')
        batch_dialogues_drop_mask = numpy.ones((max_batch_sequence_length, state['bs']), dtype='float32')

        # Fill in batch with values
        batch_dialogues[:,0:current_batch_size]  = all_dialogues[0:max_batch_sequence_length, example_indices]
        batch_dialogues_reversed[:,0:current_batch_size] = all_dialogues_reversed[0:max_batch_sequence_length, example_indices]
        batch_dialogues_mask[:,0:current_batch_size] = all_dialogues_mask[0:max_batch_sequence_length, example_indices]
        batch_dialogues_drop_mask[:,0:current_batch_size] = all_dialogues_drop_mask[0:max_batch_sequence_length, example_indices]

        if batch_idx < 10:
            print '###'
            print '###'
            print '###'

            print 'DEBUG THIS:'
            print 'batch_dialogues', batch_dialogues[:, 0]
            print 'batch_dialogues_reversed',batch_dialogues_reversed[:, 0]
            print 'max_batch_sequence_length', max_batch_sequence_length
            print 'batch_dialogues_reset_mask', batch_dialogues_reset_mask


        batch_dialogues_ran_gaussian_vectors = numpy.zeros((max_batch_sequence_length, state['bs'], model.latent_gaussian_per_utterance_dim), dtype='float32')
        batch_dialogues_ran_uniform_vectors = numpy.zeros((max_batch_sequence_length, state['bs'], model.latent_piecewise_per_utterance_dim), dtype='float32')

        for idx in range(state['bs']):
            eos_sym_list = numpy.where(batch_dialogues[:, idx] == state['eos_sym'])[0]
            if len(eos_sym_list) > 1:
                second_last_eos_sym = eos_sym_list[-2]
            else:
                print 'WARNING: batch_idx ' + str(batch_idx) + ' example index ' + str(idx) + ' does not have at least two EOS tokens!'
                print '     batch was: ', batch_dialogues[:, idx]

                if len(eos_sym_list) > 0:
                    second_last_eos_sym = eos_sym_list[-1]
                else:
                    second_last_eos_sym = 0

            batch_dialogues_ran_gaussian_vectors[second_last_eos_sym:, idx, :] = model.rng.normal(loc=0, scale=1, size=model.latent_gaussian_per_utterance_dim)
            batch_dialogues_ran_uniform_vectors[second_last_eos_sym:, idx, :] = model.rng.uniform(low=0.0, high=1.0, size=model.latent_piecewise_per_utterance_dim)

            batch_dialogues_mask[0:second_last_eos_sym+1, idx] = 0.0




        if batch_idx < 10:
            print 'batch_dialogues_mask', batch_dialogues_mask[:, 0]
            print 'batch_dialogues_ran_gaussian_vectors', batch_dialogues_ran_gaussian_vectors[:, 0, 0]
            print 'batch_dialogues_ran_gaussian_vectors [-1]', batch_dialogues_ran_gaussian_vectors[:, 0, -1]
            print 'batch_dialogues_ran_uniform_vectors', batch_dialogues_ran_uniform_vectors[:, 0, 0]
            print 'batch_dialogues_ran_uniform_vectors [-1]', batch_dialogues_ran_uniform_vectors[:, 0, -1]


        # Carry out mean-field inference:
        if args.mf_inference_steps > 0:
            mf_reset_batch()
            model.build_mf_reset_function()
            for mf_step in range(args.mf_inference_steps):


                training_cost, kl_divergence_cost_acc, kl_divergences_between_piecewise_prior_and_posterior, kl_divergences_between_gaussian_prior_and_posterior = mf_update_batch(batch_dialogues, batch_dialogues_reversed, max_batch_sequence_length, batch_dialogues_mask, batch_dialogues_reset_mask, batch_dialogues_ran_gaussian_vectors, batch_dialogues_ran_uniform_vectors, batch_dialogues_drop_mask)
                if batch_idx % 10 == 0:
                    print  '  mf_step', mf_step
                    print '     training_cost', training_cost
                    print '     kl_divergence_cost_acc', kl_divergence_cost_acc
                    print '     kl_divergences_between_gaussian_prior_and_posterior',  numpy.sum(kl_divergences_between_gaussian_prior_and_posterior)
                    print '     kl_divergences_between_piecewise_prior_and_posterior', numpy.sum(kl_divergences_between_piecewise_prior_and_posterior)



        eval_batch(batch_dialogues, batch_dialogues_reversed, max_batch_sequence_length, batch_dialogues_mask, batch_dialogues_reset_mask, batch_dialogues_ran_gaussian_vectors, batch_dialogues_ran_uniform_vectors, batch_dialogues_drop_mask)

        # Train on batch and get results
        _, c_list, _ = eval_batch(batch_dialogues, batch_dialogues_reversed, max_batch_sequence_length, batch_dialogues_mask, batch_dialogues_reset_mask, batch_dialogues_ran_gaussian_vectors, batch_dialogues_ran_uniform_vectors, batch_dialogues_drop_mask)
        c_list = c_list.reshape((state['bs'],max_batch_sequence_length-1), order=(1,0))
        c_list = numpy.sum(c_list, axis=1) / numpy.sum(batch_dialogues_mask, axis=0)
        all_dialogues_cost[example_indices] = c_list[0:current_batch_size]

        #print 'hd_input [0]', hd_input[:, 0, 0]
        #print 'hd_input [-]', hd_input[:, 0, -1]


    print('Ranking responses...')
    example_idx = -1
    ranked_all_responses_string = ''
    ranked_all_responses_costs_string = ''
    for context_idx, context in enumerate(contexts):
        if context_idx % 100 == 0:
            print '     processing context idx: ' + str(context_idx) + ' / ' + str(len(contexts))

        potential_responses = potential_responses_set[context_idx].strip().split('\t')
        potential_example_response_indices = numpy.zeros((len(potential_responses)), dtype='int32')

        for potential_response_idx, potential_response in enumerate(potential_responses):
            example_idx += 1
            potential_example_response_indices[potential_response_idx] = example_idx

        ranked_potential_example_response_indices = numpy.argsort(all_dialogues_cost[potential_example_response_indices])

        ranked_responses_costs = all_dialogues_cost[potential_example_response_indices]

        ranked_responses_string = ''
        ranked_responses_costs_string = ''
        for idx in ranked_potential_example_response_indices:
            ranked_responses_string += potential_responses[idx] + '\t'
            ranked_responses_costs_string += str(ranked_responses_costs[idx]) + '\t'

        ranked_all_responses_string += ranked_responses_string[0:len(ranked_responses_string)-1] + '\n'
        ranked_all_responses_costs_string += ranked_responses_costs_string[0:len(ranked_responses_string)-1] + '\n'

    print('Finished all computations!')
    print('Saving to file...')

    # Write to output file
    output_handle = open(args.output + '.txt', "w")
    output_handle.write(ranked_all_responses_string)
    output_handle.close()

    output_handle = open(args.output + '_Costs.txt' , "w")
    output_handle.write(ranked_all_responses_costs_string)
    output_handle.close()

    print('Saving to file finished.')
    print('All done!')

if __name__ == "__main__":
    # THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python rank_responses.py tests/models/1475190148.23_testmodel tests/data/tvalid_contexts.txt tests/data/tvalid_potential_responses.txt Out.txt --verbose

    # awk -F "\t" '{print $1}' GeneratedResponses.txt &> First_GeneratedResponses.txt
    
    main()

