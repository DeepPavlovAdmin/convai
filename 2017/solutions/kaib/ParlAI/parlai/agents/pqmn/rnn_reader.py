# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers
from .tdnn import TDNN
from .highway import Highway
import torch.nn.functional as F

import pdb


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, padding_idx_char=0):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)
        
        # Char embeddings (+1 for padding)
        #pdb.set_trace()
        if opt['add_char2word']:
            self.char_embedding = nn.Embedding(opt['vocab_size_char'],
                                               opt['embedding_dim_char'],
                                               padding_idx=padding_idx_char)

            self.char_embedding.weight = nn.Parameter(torch.Tensor(opt['vocab_size_char'],opt['embedding_dim_char']).uniform_(-1,1))

            self.TDNN = TDNN(opt)

            if opt['nLayer_Highway'] > 0 :
                self.Highway = Highway(opt['embedding_dim'] + opt['embedding_dim_TDNN'], opt['nLayer_Highway'], F.relu)

        # ...(maybe) keep them fixed
        if opt['fix_embeddings']:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # Register a buffer to (maybe) fill later for keeping *some* fixed
        if opt['tune_partial'] > 0:
            buffer_size = torch.Size((
                opt['vocab_size'] - opt['tune_partial'] - 2,
                opt['embedding_dim']
            ))
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        # Projection for attention weighted question
        if opt['use_qemb']:
            if opt['add_char2word']:
                self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'] + opt['embedding_dim_TDNN'])
            else:
                self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        if opt['add_char2word']:
            doc_input_size = opt['embedding_dim'] + opt['num_features'] + opt['embedding_dim_TDNN']
        else:
            doc_input_size = opt['embedding_dim'] + opt['num_features']

        if opt['use_qemb']:
            if opt['add_char2word']:
                doc_input_size += opt['embedding_dim'] + opt['embedding_dim_TDNN']
            else:
                doc_input_size += opt['embedding_dim']

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder
        q_input_size = opt['embedding_dim']
        if opt['add_char2word']:
            q_input_size += opt['embedding_dim_TDNN']

        self.question_rnn = layers.StackedBRNN(
            input_size=q_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Q-P matching
        opt['qp_rnn_size'] = doc_hidden_size + question_hidden_size
        if opt['qp_bottleneck']:
            opt['qp_rnn_size'] = opt['hidden_size_bottleneck']
        
        self.qp_match = layers.GatedAttentionBilinearRNN(
            x_size = doc_hidden_size,
            y_size = question_hidden_size,            
            hidden_size= opt['qp_rnn_size'],
            padding=opt['rnn_padding'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            birnn=opt['qp_birnn'],
            concat = opt['qp_concat'],
            gate=True
        )
        qp_matched_size = opt['qp_rnn_size']
        if opt['qp_birnn']:
            qp_matched_size = qp_matched_size * 2
        if opt['qp_concat']:
            qp_matched_size = qp_matched_size + doc_hidden_size        
 
        ## PP matching:             
        opt['pp_rnn_size'] = qp_matched_size * 2
        if opt['pp_bottleneck']:
            opt['pp_rnn_size'] = opt['hidden_size_bottleneck']
        
        self.pp_match = layers.GatedAttentionBilinearRNN(
            x_size = qp_matched_size,
            y_size = qp_matched_size,            
            hidden_size= opt['pp_rnn_size'],
            padding=opt['rnn_padding'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            birnn=opt['pp_birnn'],
            concat = opt['pp_concat'],
            gate=opt['pp_gate'], 
            rnn=opt['pp_rnn'],
            identity = ['pp_identity']
        )
        pp_matched_size = opt['pp_rnn_size']
        if opt['pp_birnn'] and opt['pp_rnn']:
            pp_matched_size = pp_matched_size * 2
        if opt['pp_concat']:
            pp_matched_size = pp_matched_size + qp_matched_size
                
        # Bilinear attention for span start/end
        if opt['task_QA']:
            self.start_attn = layers.BilinearSeqAttn(
                pp_matched_size,
                question_hidden_size
                )
            self.end_attn = layers.BilinearSeqAttn(
                pp_matched_size,
                question_hidden_size
                )
                                       
        # Paragraph Hierarchical Encoder
        if opt['ans_sent_predict'] :
            self.meanpoolLayer = layers.Selective_Meanpool(doc_hidden_size)
            self.sentBRNN = layers.StackedBRNN(
                input_size=pp_matched_size,
                hidden_size=opt['hidden_size_sent'],
                num_layers=opt['nLayer_Sent'],
                concat_layers=False,
                rnn_type=self.RNN_TYPES[opt['rnn_type']],
                padding=opt['rnn_padding_sent'],
            )
            self.sentseqAttn = layers.BilinearSeqAttn(
                opt['hidden_size_sent'],
                question_hidden_size,
                )


    #def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
    def forward(self, x1, x1_f, x1_mask, x2, x2_mask, x1_c=None, x2_c=None, x1_sent_mask=None, word_boundary=None):  # for this version, we do not utilize mask for char

        #pdb.set_trace()

        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d] ==>
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q] ==>
        x1_c = document char indices           [batch * len_d * max_char_per_word]
        x1_c_mask = document char padding mask [batch * len_d * max_char_per_word] --> not implemented in this version
        x2_c = question char indices           [batch * len_q * max_char_per_word]
        x2_c_mask = question char padding mask [batch * len_q * max_char_per_word] --> not implemented in this version
        """
        # Embed both document and question
        batch_size = x1.size()[0]
        doc_len = x1.size()[1]
        ques_len = x2.size()[1]
        x1_emb = self.embedding(x1) # N x Td x D
        x2_emb = self.embedding(x2) # N x Tq x D

        if self.opt['add_char2word']:
            max_wordL_d = x1_c.size()[2]
            max_wordL_q = x2_c.size()[2]
            x1_c = x1_c.view(-1, max_wordL_d)
            x2_c = x2_c.view(-1, max_wordL_q)
            x1_c_emb = self.char_embedding(x1_c)
            x2_c_emb = self.char_embedding(x2_c)
            x1_c_emb = x1_c_emb.view(batch_size,
                                     doc_len,
                                     max_wordL_d,
                                     -1)
            x2_c_emb = x2_c_emb.view(batch_size,
                                     ques_len,
                                     max_wordL_q,
                                     -1)

            # Produce char-aware word embed
            x1_cw_emb = self.TDNN(x1_c_emb)  # N x Td x sum(H)
            x2_cw_emb = self.TDNN(x2_c_emb)  # N x Tq x sum(H)

            # Merge word + char
            x1_emb = torch.cat((x1_emb, x1_cw_emb), 2)
            x2_emb = torch.cat((x2_emb, x2_cw_emb), 2)
            ###x1_mask = torch.cat([x1_mask, x1_c_mask], 2)  # For this version, we do not utilize char mask
            ###x2_mask = torch.cat([x2_mask, x2_c_mask], 2)  # For this version, we do not utilize char mask

            # Highway network
            if self.opt['nLayer_Highway'] > 0:
                [batch_size, seq_len, embed_size] = x1_emb.size()
                x1_emb = self.Highway(x1_emb.view(-1, embed_size))
                x1_emb = x1_emb.view(batch_size, -1, embed_size)

                [batch_size, seq_len, embed_size] = x2_emb.size()
                x2_emb = self.Highway(x2_emb.view(-1, embed_size))
                x2_emb = x2_emb.view(batch_size, -1, embed_size)
        else:
            if (('x1_c' in locals()) and ('x2_c' in locals())):
                #pdb.set_trace()
                x1_sent_mask = x1_c
                word_boundary = x2_c

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'], training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'], training=self.training)

        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input = torch.cat([x1_emb, x2_weighted_emb, x1_f], 2)
        else:
            drnn_input = torch.cat([x1_emb, x1_f], 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        # Encode question with RNN
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        
        # QP matching
        qp_matched_doc = self.qp_match(doc_hiddens, x1_mask, question_hiddens, x2_mask)
        
        # PP matching
        if not qp_matched_doc.is_contiguous():
            qp_matched_doc = qp_matched_doc.contiguous()
            
        pp_matched_doc = self.pp_match(qp_matched_doc, x1_mask, qp_matched_doc, x1_mask)
        #print(pp_matched_doc.size())
        #pdb.set_trace()
        
        # Merge question hiddens
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)


        return_list = []
        # Predict start and end positions
        if self.opt['task_QA']:
            start_scores = self.start_attn(pp_matched_doc, question_hidden, x1_mask)
            end_scores = self.end_attn(pp_matched_doc, question_hidden, x1_mask)
            return_list = return_list + [start_scores, end_scores]

        # Pooling , currently no multi-task learning
        if self.opt['ans_sent_predict']:
            sent_hiddens = self.meanpoolLayer(pp_matched_doc, word_boundary)
            if self.opt['nLayer_Sent'] > 0:
                sent_hiddens = self.sentBRNN(sent_hiddens, x1_sent_mask)

            sent_scores = self.sentseqAttn(sent_hiddens, question_hidden, x1_sent_mask)
            return_list = return_list + [sent_scores]

        return return_list
    
    
    
