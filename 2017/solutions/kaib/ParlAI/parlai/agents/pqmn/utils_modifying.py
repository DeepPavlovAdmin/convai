# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import time
import unicodedata
from collections import Counter
import spacy
import numpy as np
import pdb

NLP = spacy.load('en')
pos_list = [
    'DET', 'ADP', 'PART', 'ADJ', 'PUNCT', 'INTJ', 'NOUN', 'ADV', 'X', 'PRON',
    'PROPN', 'VERB', 'CONJ', 'SPACE', 'NUM', 'SYM', 'CCONJ'
]
ner_list = [
    'QUANTITY', 'PRODUCT', 'EVENT', 'FACILITY', 'NORP', 'TIME', 'LANGUAGE',
    'ORG', 'DATE', 'CARDINAL', 'PERSON', 'ORDINAL', 'LOC', 'PERCENT', 'MONEY',
    'WORK_OF_ART', 'GPE', 'FAC', 'LAW'
]
pos_dict = {i: pos_list.index(i)/len(pos_list) for i in pos_list}
ner_dict = {i: ner_list.index(i)/len(ner_list) for i in ner_list}

# ------------------------------------------------------------------------------
# Data/model utilities.
# ------------------------------------------------------------------------------


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors."""
    embeddings = torch.Tensor(len(word_dict), opt['embedding_dim'])
    embeddings.normal_(0, 1)

    # Fill in embeddings
    if not opt.get('embedding_file'):
        raise RuntimeError('Tried to load embeddings with no embedding file.')
    with open(opt['embedding_file']) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            assert(len(parsed) == opt['embedding_dim'] + 1)
            w = normalize_text(parsed[0])
            if w in word_dict:
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                embeddings[word_dict[w]].copy_(vec)

    # Zero NULL token
    embeddings[word_dict['<NULL>']].fill_(0)

    return embeddings


def build_feature_dict(opt):
    """Make mapping of feature option to feature index."""
    feature_dict = {}
    if opt['use_in_question']:
        feature_dict['in_question'] = len(feature_dict)
        feature_dict['in_question_uncased'] = len(feature_dict)
    if opt['use_tf']:
        feature_dict['tf'] = len(feature_dict)
    if opt['use_ner']:
        feature_dict['ner_type'] = len(feature_dict)
    if opt['use_pos']:
        feature_dict['pos_type'] = len(feature_dict)
    if opt['use_time'] > 0:
        for i in range(opt['use_time'] - 1):
            feature_dict['time=T%d' % (i + 1)] = len(feature_dict)
        feature_dict['time>=T%d' % opt['use_time']] = len(feature_dict)
    return feature_dict


# ------------------------------------------------------------------------------
# Torchified input utilities.
# ------------------------------------------------------------------------------


#def vectorize(opt, ex, word_dict, feature_dict):
def vectorize(opt, ex, word_dict, char_dict, feature_dict):
    """Turn tokenized text inputs into feature vectors."""
    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

"""
    # Index words with character-level tensor
    T_doc = len(ex['document'])
    T_ques = len(ex['question'])

    print ("T_docs = ", T_doc, ", T_ques = ", T_ques)

    document_char = np.array(
        [list(map(encode_characters(w, char_dict, opt['max_word_len'], 0))) for w in ex['document']])  # 0 = NULLIdx
    question_char = np.array(
        [list(map(encode_characters(w, char_dict, opt['max_word_len'], 0))) for w in ex['question']])  # 0 = NULLIdx

    document_char=None
    question_char=None
    if opt['add_char2word']:
        document_char = np.array(
            [encode_characters(w, char_dict, opt['max_word_len'], 0) for w in ex['document']])  # 0 = NULLIdx
        question_char = np.array(
            [encode_characters(w, char_dict, opt['max_word_len'], 0) for w in ex['question']])  # 0 = NULLIdx

        document_char = [torch.from_numpy(var) for var in document_char]
        document_char = [var.long() for var in document_char]

        question_char = [torch.from_numpy(var) for var in question_char]
        question_char = [var.long() for var in question_char]
"""
   # Create extra features vector
    features = torch.zeros(len(ex['document']), len(feature_dict))

    # f_{exact_match}
    if opt['use_in_question']:
        q_words_cased = set([w for w in ex['question']])
        q_words_uncased = set([w.lower() for w in ex['question']])
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0

    # f_{tf}
    if opt['use_tf']:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    if opt['use_time'] > 0:
        # Counting from the end, each (full-stop terminated) sentence gets
        # its own time identitfier.
        sent_idx = 0
        def _full_stop(w):
            return w in {'.', '?', '!'}
        for i, w in reversed(list(enumerate(ex['document']))):
            sent_idx  = sent_idx + 1 if _full_stop(w) else max(sent_idx, 1)
            if sent_idx < opt['use_time']:
                features[i][feature_dict['time=T%d' % sent_idx]] = 1.0
            else:
                features[i][feature_dict['time>=T%d' % opt['use_time']]] = 1.0

"""
    # Maybe return without target
    if ex['target'] is None:
        if opt['add_char2word']:
            return document, features, question, document_char, question_char
        else:
            return document, features, question
"""
    # Maybe return without target
    if ex['target'] is None:
        return document, features, question

    # ...or with target
    start = torch.LongTensor(1).fill_(ex['target'][0])
    end = torch.LongTensor(1).fill_(ex['target'][1])
    
    return document, features, question, start, end
    
"""
    # Answer Sentence Prediction
    if opt['ans_sent_predict']:
        word_boundary = np.array([w for w in ex['word_idx']])
        answer_sent = ex['answer_sent']
        if opt['add_char2word']:
            return document, features, question, document_char, question_char, word_boundary, answer_sent, start, end  # document_char, question_char : np.array
        else:
            return document, features, question, word_boundary, answer_sent, start, end

    #return document, features, question, start, end
    if opt['add_char2word']:
        return document, features, question, document_char, question_char, start, end  # document_char, question_char : np.array
    else:
        return document, features, question, start, end
"""


def batchify(batch, null=0, cuda=False):
#def batchify(batch, null=0, max_word_len=15, NULLWORD_Idx_in_char=99, cuda=False, use_char=False, sent_predict=False):
    """Collate inputs into batches."""
""
    NUM_INPUTS = 3 # doc(word) + feature + ques(word)
    #NUM_INPUTS = 5  # doc(word) + feature + ques(word) + doc(char) + ques(char)

    n_word_idx=0
    n_sent_label=0
    if sent_predict:
        NUM_INPUTS += 2
        n_word_idx+=3
        n_sent_label+=4
    if use_char:
        NUM_INPUTS += 2
        n_doc_char=3
        n_ques_char=4
        n_word_idx+=2
        n_sent_label+=2
        
    NUM_TARGETS = 2
    NUM_EXTRA = 2

    # Get elements
    #pdb.set_trace()
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    if use_char:
        docs_char = [ex[n_doc_char] for ex in batch]
        ques_char = [ex[n_ques_char] for ex in batch]
    if sent_predict:
        word_idx = [ex[n_word_idx] for ex in batch]
        sent_label= [ex[n_sent_label] for ex in batch]

    text = [ex[-2] for ex in batch]
    spans = [ex[-1] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).fill_(null)
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        x1_f[i, :d.size(0)].copy_(features[i])

    # pdb.set_trace()

    # x1_sent_mask
    if sent_predict:
        #print(word_idx)
        maxSent = len(max(word_idx, key=len))
        lenSent = [len(l) for l in word_idx]
        x1_sent_mask = torch.ByteTensor(len(docs),maxSent ).fill_(1)
        for n in range(len(docs)):
            #print(lenSent[n])
            x1_sent_mask[n, :lenSent[n]].fill_(0)


    # Batch document with character level encoding
    if use_char:
        x1_c = torch.LongTensor(len(docs), max_length, max_word_len).fill_(null)
    # Batch questions
        for i in range(len(docs_char)):  # Iterate over documents
            for j in range(len(docs_char[i])):  # Iterate over words
                #print('(i,j) = (', i, ', ', j, ')')
                #print(docs_char[i][j].size())
                x1_c[i, j, :docs_char[i][j].size(0)].copy_(docs_char[i][j])
            if(len(docs_char[i]) < max_length):
                x1_c[i, len(docs_char[i]):, 0].fill_(NULLWORD_Idx_in_char)  # fill <NULL_WORD>

    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).fill_(null)
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)


    # Batch question with character level encoding
    if use_char:
        x2_c = torch.LongTensor(len(questions), max_length, max_word_len).fill_(null)
        for i in range(len(ques_char)):  # Iterate over questions
            for j in range(len(ques_char[i])): # Iterate over word
                x2_c[i, j, :ques_char[i][j].size(0)].copy_(ques_char[i][j])
            if(len(ques_char[i]) < max_length):
                x2_c[i, len(ques_char[i]):, 0].fill_(NULLWORD_Idx_in_char)

    #pdb.set_trace()

    # Pin memory if cuda
    if cuda:
        x1 = x1.pin_memory()
        x1_f = x1_f.pin_memory()
        x1_mask = x1_mask.pin_memory()
        x2 = x2.pin_memory()
        x2_mask = x2_mask.pin_memory()
        if use_char:
            x1_c = x1_c.pin_memory()
            x2_c = x2_c.pin_memory()
        if sent_predict:
            x1_sent_mask = x1_sent_mask.pin_memory()

    # Maybe return without targets



    #pdb.set_trace()
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        #return x1, x1_f, x1_mask, x2, x2_mask, text, spans
        return_list = [x1, x1_f, x1_mask, x2, x2_mask]
        if use_char:
            return_list = return_list + [x1_c, x2_c]
        if sent_predict:
            return_list = return_list + [x1_sent_mask, word_idx, sent_label]

        return_list = return_list + [text, spans]
        return return_list

        #return x1, x1_f, x1_mask, x2, x2_mask, x1_c, x2_c, text, spans
        #else:
    #            return x1, x1_f, x1_mask, x2, x2_mask, text, spans

    # ...Otherwise add targets
    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        y_s = torch.cat([ex[NUM_INPUTS] for ex in batch])
        y_e = torch.cat([ex[NUM_INPUTS+1] for ex in batch])

        return_list = [x1, x1_f, x1_mask, x2, x2_mask]
        if use_char:
            return_list = return_list + [x1_c, x2_c]
        if sent_predict:
            return_list = return_list + [x1_sent_mask, word_idx, sent_label]
        return_list = return_list + [y_s, y_e, text, spans]

        return return_list

        #if use_char:
        #    return x1, x1_f, x1_mask, x2, x2_mask, x1_c, x2_c, y_s, y_e, text, spans
        #else:
        #    return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, text, spans

    # ...Otherwise wrong number of inputs
    raise RuntimeError('Wrong number of inputs per batch')


# ------------------------------------------------------------------------------
# Character processing.
# ------------------------------------------------------------------------------
#def encode_characters(characters, dict_char, max_word_len, NULLIdx, flag_debug=False):
def encode_characters(characters, dict_char, max_word_len, NULLIdx):
    #pdb.set_trace()
    word_len = len(characters)
    if(word_len > max_word_len):
        characters = characters[:max_word_len]  # Make sure characters length smaller than max_word_len

    #assert(word_len <= max_word_len)
    to_add = max_word_len - word_len
    characters_idx = [dict_char[i] for i in characters] + to_add * [NULLIdx] # right padding
    #print('before')
    #print(type(characters_idx))

    characters_idx = np.asarray(characters_idx)

    #print('after')
    #print(type(characters_idx))


    #if(flag_debug):
     #   print(characters_idx)
     
    return characters_idx

# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
