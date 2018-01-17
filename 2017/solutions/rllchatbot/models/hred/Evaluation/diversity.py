__docformat__ = 'restructedtext en'
__authors__ = ("Iulian Vlad Serban")

import operator
import numpy as np
import argparse
import cPickle
import math

def tf(fileone, filetwo, w2v):
    r1 = f1.readlines()
    r2 = f2.readlines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predicted', help="predicted text file, one example per line")
    parser.add_argument('dictionary', help="dictionary pickle file")

    args = parser.parse_args()

    print "loading dictionary file..."
    raw_dict = cPickle.load(open(args.dictionary, 'r'))
    word_freq = dict([(tok, freq) for tok, _, freq, _ in raw_dict])
    document_freq = dict([(tok, df) for tok, _, _, df in raw_dict])
    document_count = np.max(document_freq.values())
    print 'document_count', document_count

    print "precomputing inverse-document-frequency values..."
    inverse_document_freq = {}
    for word in document_freq.keys():
        inverse_document_freq[word] = math.log(float(document_count)/max(1.0, float(document_freq[word])))

    print "precomputing bag-of-word probabilities..."
    total_word_count = 0
    for word in word_freq.keys():
        total_word_count += word_freq[word]

    word_logprob = {}
    for word in word_freq.keys():
        word_logprob[word] =  math.log(max(1.0, float(word_freq[word]))/float(total_word_count), 2)


    responses_in_total = 0.0
    responses_less_than_15 = 0.0



    print "loading model utterances..."
    model_utterances = open(args.predicted, 'r').readlines()

    print "computing TF and TF-IDF scores..."

    wordpos_word_count = {}
    wordpos_to_idf = {}
    wordpos_to_log_prob = {}

    model_word_count = 0.0
    model_word_log_prob_sum = 0.0


    count_words = {}    
    unique_words = {}
    unique_words_per_response = 0.0

    for idx in range(len(model_utterances)):
        if idx % 1000 == 0:
            print '   example ', idx


        model_utterance = model_utterances[idx].strip()

        model_utterance_words = model_utterance.split()
        unique_words_per_response += len(set(model_utterance_words))

        # Compute word position statistics for model generated response
        for wrd_idx, word in enumerate(model_utterance_words):
            wordpos_word_count[wrd_idx] = wordpos_word_count.get(wrd_idx, 0.0) + 1.0
            wordpos_to_idf[wrd_idx] = wordpos_to_idf.get(wrd_idx, 0.0) + inverse_document_freq.get(word, 1.0)
            wordpos_to_log_prob[wrd_idx] = wordpos_to_log_prob.get(wrd_idx, 0.0) + word_logprob.get(word, 0.0)

            model_word_count += 1.0
            model_word_log_prob_sum += word_logprob.get(word, 0.0)

            count_words[word] = count_words.get(word, 0) + 1
            unique_words[word] = True

        responses_in_total += 1.0
        if len(model_utterance_words) <= 15:
            responses_less_than_15 += 1.0

    print("Word Position versus Word IDF in Model Responses")
    for pos in range(0, 40):
        print '    ' + str(pos) + ' ' + str(wordpos_to_idf.get(pos, 0.0) / wordpos_word_count.get(pos, 1.0))

    print("")

    print("Word Position versus (BoW) Word Log-Likelihood in Model Responses")
    for pos in range(0, 40):
        print '    ' + str(pos) + ' ' + str(-wordpos_to_log_prob.get(pos, 0.0) / wordpos_word_count.get(pos, 1.0))

    print 'responses_less_than_15', str(responses_less_than_15/responses_in_total)


    print 'Average model per-word entropy', str(-model_word_log_prob_sum/model_word_count)

    print 'Unique words', len(set(unique_words.keys()))

    print 'Unique words per response', unique_words_per_response / float(len(model_utterances))
    #assert False




    print 'Word Count by Rank'
    sorted_count_words = sorted(count_words.items(), key=operator.itemgetter(1))
    for i in reversed(range(len(count_words))):
        print sorted_count_words[i][1]

