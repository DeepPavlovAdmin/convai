# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import pdb

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

class Selective_Meanpool(nn.Module):
    def __init__(self, input_size):
        super(Selective_Meanpool, self).__init__()
        self.input_size = input_size

    def forward(self, x, word_end):
        """Mean pool across word boundary."""

        # x : N x Tword x H (32 x 500 x 768)
        # word_end : word end index of each paragraph, list

        nBatch = len(word_end)
        maxSent = len(max(word_end, key=len))

        outputs = []

        #pdb.set_trace()
        for n in range(nBatch):
            outputs_batch = []
            startend = np.insert(word_end[n], 0, -1)
            nSentence = len(startend)-1

            #start_idx = Variable(torch.from_numpy(startend[:-1]) + 1)  # Variable,
            #end_idx = Variable(torch.from_numpy(startend[1:]) )         # Variable

            start_idx = startend[:-1] + 1  # numpy.array
            end_idx = startend[1:] # numpy.array

            for s in range(nSentence):
                end_idx_real = end_idx[s]+1
                if end_idx_real < 0 :
                    end_idx_real = x.size()[1]
                meanpool_idx = torch.from_numpy(np.arange(start_idx[s], end_idx_real))
                meanpool_idx = Variable(meanpool_idx.cuda(async=True))
                outputs_batch.append(torch.mean(x[n,:, :].index_select(0, meanpool_idx),0))

            if nSentence < maxSent:  # zero tensor padding
                outputs_batch.append(Variable(torch.zeros(maxSent-nSentence, x.size()[-1]).cuda(async=True), requires_grad=False))

            outputs_batch_tensor = torch.cat(outputs_batch, 0)
            outputs.append(outputs_batch_tensor)

        #pdb.set_trace()
        output = torch.stack(outputs, 0)

        return output

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy)
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


class GatedAttentionBilinearRNN(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.  --- eq(4) in r-net
    (X=passage u^P,  Y=Question u^Q)    
    * alpha^t_i = softmax(u^Q_j * Wu * u^P_i)
    * c_t = sum(alpha^t_i * u^Q_i) for i in X
    * gated[u^P_t, c_t] = sigmoid(W_g * [u^P_t, c_t])
    * v^P_t = RNN(v^P_(t-1), gated[u^P_t, c_t])
    """    
        
    def __init__(self, x_size, y_size, hidden_size,
                 rnn_type=nn.LSTM,
                 gate=True, padding = False,
                 birnn=False, identity=False, concat=False, rnn=True):
        super(GatedAttentionBilinearRNN, self).__init__()
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.padding = padding
        self.concat_layers = concat
        
        if not identity:
            self.linear = nn.Linear(y_size, x_size, bias=False)
        else:
            self.linear = None
        
        self.gate = gate
        if self.gate:
            self.gate_layer = nn.Sequential(
                nn.Linear(y_size + x_size, 1, bias=False ),  # the 2nd hidden_size can be different from 'hidden_size'
                nn.Sigmoid())
        
        if not (hidden_size == (x_size+y_size)):
            self.bottleneck_layer = nn.Linear(y_size + x_size, hidden_size)
            input_size = hidden_size
        else:
            self.bottleneck_layer = None
            input_size = y_size + x_size
        
        self.rnn = rnn 
        if self.rnn:
            self.rnns = nn.ModuleList()
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1, bidirectional=birnn))
        
        self.alpha = None
        
    def forward(self,  x, x_mask, y, y_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask, y, y_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask, y, y_mask)
        
        # We don't care.
        return self._forward_unpadded(x, x_mask, y, y_mask)
    
    def get_alpha(self):
        return self.alpha
         
    def _gated_attended_input(self, x, x_mask, y, y_mask):
        nbatch = x.size(0) #(batch, seq_len, input_size)
        x_len = x.size(1)
        y_len = y.size(1)        
        x_size = x.size(2)
        y_size = y.size(2)
        
        #tic = time.time()    

        # Attention
        # * alpha^t_i = softmax(tanh( u^Q_j * W * u^P_i ))
        # * c_t = sum(alpha^t_i * u^Q_i) for i in X
        #pdb.set_trace()
        Wy = self.linear(y.view(-1, y_size)).view(-1, y_len, x_size) if self.linear is not None else y
        xWy = x.bmm(Wy.transpose(1,2))
        xWy.data.masked_fill_(y_mask.data.unsqueeze(1).expand_as(xWy), -float('inf'))
        alpha = F.softmax(xWy.view(-1, y_len))

        alpha = alpha.view(nbatch, x_len, y_len)
        attend_y = alpha.bmm(y)
        self.alpha = alpha
        
        attend_y.data.masked_fill_(x_mask.unsqueeze(2).expand_as(attend_y).data, 0) ## comment out?
        rnn_input = torch.cat((x, attend_y), 2)

        # Gate: gated[u^P_t, c_t] = sigmoid(W_g * [u^P_t, c_t])
        if self.gate:
            gate = self.gate_layer(rnn_input.view(-1, rnn_input.size(2))).view(nbatch, x_len, 1).expand_as(rnn_input) #1, 1, rnn_input.size(2))
            rnn_input = gate.mul(rnn_input)
        
        # 128*3 *2= 1536 ==> too large as an RNN input? then insert a bottle neck layer 
        rnn_input = self.bottleneck_layer(rnn_input.view(-1, rnn_input.size(2))).view(nbatch, x_len, -1) if self.bottleneck_layer is not None else rnn_input        
        
        return rnn_input
        
    def _forward_unpadded(self,  x, x_mask, y, y_mask):
        """Faster encoding that ignores any padding."""
        # Encode all layers
        output = self._gated_attended_input(x, x_mask, y, y_mask)
        if self.rnn:
            outputs = [output] 
            for i in range(self.num_layers): ## self.num_layers == 1
                # RNN: v^P_t = RNN(v^P_(t-1), gated[u^P_t, c_t])   
                rnn_output = self.rnns[i](outputs[-1].transpose(0,1))[0] # batch_first = False
                outputs.append(rnn_output)
                output = outputs[1].transpose(0,1)
            
       # Concat hidden layers
        if self.concat_layers:
            output = torch.cat((output, x), 2)
            
        return output

    def _forward_padded(self,  x, x_mask, y, y_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)       

        input = self._gated_attended_input(x, x_mask, y, y_mask)
        
        if self.rnn:
            # Sort x
            input = input.index_select(0, idx_sort)
    
            # Transpose batch and sequence dims
            input = input.transpose(0, 1)
    
            # Pack it up
            rnn_input = nn.utils.rnn.pack_padded_sequence(input, lengths)
    
            # Encode all layers
            outputs = [rnn_input]
            for i in range(self.num_layers):
                rnn_input = outputs[-1]
                outputs.append(self.rnns[i](rnn_input)[0])
    
            # Unpack everything
            for i, o in enumerate(outputs[1:], 1):
                outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

            # Transpose and unsort
            output = outputs[1].transpose(0, 1)
            output = output.index_select(0, idx_unsort)
        else:
            output = input
        
        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat((output, x), 2)
            
        return output

# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
