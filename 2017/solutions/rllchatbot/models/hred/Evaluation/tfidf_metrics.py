"""
TF-IDF evaluation metrics for dialogue.

This method implements two evaluation metrics:
1) term-frequency cosine similarity between target utterance and model utterance
2) term-frequency inverse-document-frequency cosine similarity between target utterance and model utterance, where each dialogue corresponds to one document and where the inverse-document-frequency is given by ...

We believe that these metrics are suitable for evaluating how well dialogue systems stay on topic.

Example run:

    python tfidf_metrics.py path_to_ground_truth.txt path_to_predictions.txt path_to_dictionary.pkl

The script assumes one example per line (e.g. one dialogue or one sentence per line), where line n in 'path_to_ground_truth.txt' matches that of line n in 'path_to_predictions.txt'.

"""
__docformat__ = 'restructedtext en'
__authors__ = ("Iulian Vlad Serban")

import numpy as np
import argparse
import cPickle
import math

def tf(fileone, filetwo, w2v):
    r1 = f1.readlines()
    r2 = f2.readlines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth', help="ground truth text file, one example per line")
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


    print "loading ground truth utterances..."
    gt_utterances = open(args.ground_truth, 'r').readlines()
    print "loading model utterances..."
    model_utterances = open(args.predicted, 'r').readlines()

    assert len(gt_utterances) == len(model_utterances)

    print "computing TF and TF-IDF scores..."

    tf_scores = []
    tfidf_scores = []

    wordpos_word_count = {}
    wordpos_to_idf = {}
    wordpos_to_log_prob = {}

    model_word_count = 0.0
    model_word_log_prob_sum = 0.0

    unique_words = {}


    for idx in range(len(gt_utterances)):
        if idx % 1000 == 0:
            print '   example ', idx

        gt_utterance = gt_utterances[idx].strip()
        model_utterance = model_utterances[idx].strip()

        # We don't count empty targets,
        # since these would always give cosine similarity one with empty responses!
        if len(gt_utterance) == 0:
            continue

        if len(model_utterance) == 0:
            tf_scores.append(0.0)
            tfidf_scores.append(0.0)

        gt_utterance_words = gt_utterance.split()
        model_utterance_words = model_utterance.split()

        # Compute target vector norms
        tf_target_norm = 0.0
        tfidf_target_norm = 0.0
        for wrd_idx, word in enumerate(set(gt_utterance_words)):
            cnt_target = gt_utterance_words.count(word)
            tf_target_norm += cnt_target**2
            tfidf_target_norm += (cnt_target*inverse_document_freq.get(word, 1.0))**2

        # Compute dot product between vectors and model vector norms
        tf_model_norm = 0.0
        tfidf_model_norm = 0.0
        tf_dot = 0.0
        tfidf_dot = 0.0
        for wrd_idx, word in enumerate(set(model_utterance_words)):
            unique_words[word] = True

            cnt_model = gt_utterance_words.count(word)
            tf_model_norm += cnt_model**2
            tfidf_model_norm += (cnt_model*inverse_document_freq.get(word, 1.0))**2

            if word in gt_utterance_words:
                cnt_target = gt_utterance_words.count(word)
                tf_dot += cnt_target * cnt_model
                tfidf_dot += cnt_target * cnt_model * (inverse_document_freq.get(word, 1.0))**2

        # Compute TF score
        if (tf_target_norm > 0.0000001) and (tf_model_norm > 0.0000001):
            tf_score = tf_dot / (np.sqrt(tf_target_norm) * np.sqrt(tf_model_norm))
        else:
            tf_score = 0.0

        # Compute TF-IDF score
        if (tfidf_target_norm > 0.0000001) and (tfidf_model_norm > 0.0000001):
            tfidf_score = tfidf_dot / (np.sqrt(tfidf_target_norm) * np.sqrt(tfidf_model_norm))
        else:
            tfidf_score = 0.0

        tf_scores.append(tf_score)
        tfidf_scores.append(tfidf_score)

        # Compute word position statistics for model generated response
        for wrd_idx, word in enumerate(model_utterance_words):
            wordpos_word_count[wrd_idx] = wordpos_word_count.get(wrd_idx, 0.0) + 1.0
            wordpos_to_idf[wrd_idx] = wordpos_to_idf.get(wrd_idx, 0.0) + inverse_document_freq.get(word, 1.0)
            wordpos_to_log_prob[wrd_idx] = wordpos_to_log_prob.get(wrd_idx, 0.0) + word_logprob.get(word, 0.0)

            model_word_count += 1.0
            model_word_log_prob_sum += word_logprob.get(word, 0.0)

        responses_in_total += 1.0
        if len(model_utterance_words) <= 15:
            responses_less_than_15 += 1.0


    tf_scores = np.asarray(tf_scores)
    tfidf_scores = np.asarray(tfidf_scores)

    print("TF Score: %f +/- %f " %(np.mean(tf_scores), np.std(tf_scores)))
    print("TF-IDF Score: %f +/- %f " %(np.mean(tfidf_scores), np.std(tfidf_scores)))
    print("")
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
