# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Hwaran Lee, KAIST: 2017-present

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
#from .beam import Beam
from .beam_diverse import Beam

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch

import os
import random, math
import pdb

class Seq2seqV2Agent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    For more information, see Sequence to Sequence Learning with Neural
    Networks `(Sutskever et al. 2014) <https://arxiv.org/abs/1409.3215>`_.
    """

    OPTIM_OPTS = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
    }

    ENC_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-emb', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learning_rate', type=float, default=0.5,
                           help='learning rate')
        agent.add_argument('-wd', '--weight_decay', type=float, default=0,
                           help='weight decay')
        agent.add_argument('-dr', '--dropout', type=float, default=0.2,
                           help='dropout rate')
        agent.add_argument('-att', '--attention', default=False, type='bool',
                           help='if True, use attention')
        agent.add_argument('-attType', '--attn-type', default='general',
                           choices=['general', 'concat', 'dot'],
                           help='general=bilinear dotproduct, concat=bahdanau\'s implemenation')        
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the mean score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type='bool', default=True,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length and to '
                           'be similar in length to one another by throwing '
                           'away extra tokens. This reduces the total amount '
                           'of padding in the batches.')
        agent.add_argument('-enc', '--encoder', default='gru',
                           choices=Seq2seqV2Agent.ENC_OPTS.keys(),
                           help='Choose between different encoder modules.')
        agent.add_argument('-bi', '--bi-encoder', default=True, type='bool',
                           help='Bidirection of encoder')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'] + list(Seq2seqV2Agent.ENC_OPTS.keys()),
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights.')
        agent.add_argument('-opt', '--optimizer', default='sgd',
                           choices=Seq2seqV2Agent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-gradClip', '--grad-clip', type=float, default=-1,
                       help='gradient clip, default = -1 (no clipping)')
        agent.add_argument('-epi', '--episode-concat', type='bool', default=False,
                       help='If multiple observations are from the same episode, concatenate them.')
        agent.add_argument('--beam_size', type=int, default=0,
                           help='Beam size for beam search (only for generation mode) \n For Greedy search set 0')
        agent.add_argument('--max_seq_len', type=int, default=50,
                           help='The maximum sequence length, default = 50')
        
                
    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.

            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, self.states = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.override_opt(new_opt)
                
            self.dict = DictionaryAgent(opt)
            self.id = 'Seq2Seq'
            # we use START markers to start our output
            self.START = self.dict.start_token
            self.START_TENSOR = torch.LongTensor(self.dict.parse(self.START))
            # we use END markers to end our output
            self.END = self.dict.end_token
            self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict.txt2vec(self.dict.null_token)[0]

            # store important params directly
            hsz = opt['hiddensize']
            emb = opt['embeddingsize']
            self.hidden_size = hsz
            self.emb_size = emb
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learning_rate']
            self.rank = opt['rank_candidates']
            self.longest_label = 1
            self.truncate = opt['truncate']
            self.attention = opt['attention']

            # set up tensors
            if self.opt['bi_encoder']:   
                self.zeros = torch.zeros(2*self.num_layers, 1, hsz)
            else:
                self.zeros = torch.zeros(self.num_layers, 1, hsz)
                
            self.zeros_dec = torch.zeros(self.num_layers, 1, hsz)

            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            self.cands = torch.LongTensor(1, 1, 1)
            self.cand_scores = torch.FloatTensor(1)
            self.cand_lengths = torch.LongTensor(1)

            # set up modules
            self.criterion = nn.NLLLoss(size_average = False, ignore_index = 0)
            
            # lookup table stores word embeddings
            self.lt = nn.Embedding(len(self.dict), emb,
                                   padding_idx=self.NULL_IDX)
                                   #scale_grad_by_freq=True)
            # encoder captures the input text
            enc_class = Seq2seqV2Agent.ENC_OPTS[opt['encoder']]
            self.encoder = enc_class(emb, hsz, opt['numlayers'], bidirectional=opt['bi_encoder'], dropout = opt['dropout'])
            # decoder produces our output states
            
            #if opt['decoder'] == 'shared':
            #    self.decoder = self.encoder
            dec_isz = emb+hsz
            if opt['bi_encoder']:
                dec_isz += hsz
            
            if opt['decoder'] == 'same':
                self.decoder = enc_class(dec_isz, hsz, opt['numlayers'], dropout = opt['dropout'])
            else:
                dec_class = Seq2seqV2Agent.ENC_OPTS[opt['decoder']]
                self.decoder = dec_class(dec_isz, hsz, opt['numlayers'], dropout = opt['dropout'])
            
            # linear layer helps us produce outputs from final decoder state
            self.h2o = nn.Linear(hsz, len(self.dict))
            # droput on the linear layer helps us generalize
            self.dropout = nn.Dropout(opt['dropout'])

            self.use_attention = False
            self.attn = None
            # if attention is greater than 0, set up additional members
            if self.attention:
                self.use_attention = True
                self.att_type = opt['attn_type']
                input_size = hsz                
                if opt['bi_encoder']:
                        input_size += hsz
                        
                if self.att_type == 'concat':
                    self.attn = nn.Linear(input_size+hsz, 1, bias=False)
                elif self.att_type == 'dot':
                    assert not opt['bi_encoder'] 
                elif self.att_type == 'general':
                    self.attn = nn.Linear(hsz, input_size, bias=False)

            # initialization
            """
                getattr(self, 'lt').weight.data.uniform_(-0.1, 0.1)
                for module in {'encoder', 'decoder'}:
                    for weight in getattr(self, module).parameters():
                        weight.data.normal_(0, 0.05)
                    #for bias in getattr(self, module).parameters():
                    #    bias.data.fill_(0)
                        
                for module in {'h2o', 'attn'}:
                    if hasattr(self, module):
                        getattr(self, module).weight.data.normal_(0, 0.01)
                        #getattr(self, module).bias.data.fill_(0)
            """
            
            
            # set up optims for each module
            self.lr = opt['learning_rate']
            self.wd = opt['weight_decay'] is not 0

            optim_class = Seq2seqV2Agent.OPTIM_OPTS[opt['optimizer']]
            self.optims = {
                'lt': optim_class(self.lt.parameters(), lr=self.lr),
                'encoder': optim_class(self.encoder.parameters(), lr=self.lr),
                'decoder': optim_class(self.decoder.parameters(), lr=self.lr),
                'h2o': optim_class(self.h2o.parameters(), lr=self.lr, weight_decay=self.wd),                
            }
            if self.attention and self.attn is not None:
                self.optims.update({'attn': optim_class(self.attn.parameters(), lr=self.lr, weight_decay=self.wd)})
            
            if hasattr(self, 'states'):
                # set loaded states if applicable
                self.set_states(self.states)

            if self.use_cuda:
                self.cuda()

            self.loss = 0
            self.ndata = 0
            self.loss_valid = 0
            self.ndata_valid = 0
            
            if opt['beam_size'] > 0:
                self.beamsize = opt['beam_size'] 
            
        self.episode_concat = opt['episode_concat']
        self.training = True
        self.generating = False
        self.local_human = False
        
        if opt.get('max_seq_len') is not None:
            self.max_seq_len = opt['max_seq_len']
        else:
            self.max_seq_len = opt['max_seq_len'] = 50
        self.reset()
    
    def set_lrate(self,lr):
        self.lr = lr
        for key in self.optims:
            self.optims[key].param_groups[0]['lr'] = self.lr
    
    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'decoder'}
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        return self.dict.vec2txt(vec)

    def cuda(self):
        """Push parameters to the GPU."""
        self.START_TENSOR = self.START_TENSOR.cuda(async=True)
        self.END_TENSOR = self.END_TENSOR.cuda(async=True)
        self.zeros = self.zeros.cuda(async=True)
        self.zeros_dec = self.zeros_dec.cuda(async=True)
        self.xs = self.xs.cuda(async=True)
        self.ys = self.ys.cuda(async=True)
        self.cands = self.cands.cuda(async=True)
        self.cand_scores = self.cand_scores.cuda(async=True)
        self.cand_lengths = self.cand_lengths.cuda(async=True)
        self.criterion.cuda()
        self.lt.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.h2o.cuda()
        self.dropout.cuda()
        if self.use_attention:
            self.attn.cuda()
            
    def hidden_to_idx(self, hidden, dropout=False):
        """Convert hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        if dropout:
            hidden = self.dropout(hidden) # dropout over the last hidden
        scores = self.h2o(hidden)
        scores = F.log_softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores

    def zero_grad(self):
        """Zero out optimizers."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True
    
    def preprocess(self, reply_text):
        # preprocess for opensub
        reply_text = reply_text.replace('\\n', '\n') ## TODO: pre-processing
        reply_text=reply_text.replace("'m", " 'm")
        reply_text=reply_text.replace("'ve", " 've")
        reply_text=reply_text.replace("'s", " 's")
        reply_text=reply_text.replace("'t", " 't")
        reply_text=reply_text.replace("'il", " 'il")
        reply_text=reply_text.replace("'d", " 'd")
        reply_text=reply_text.replace("'re", " 're")        
        reply_text = reply_text.lower().strip()
        
        return reply_text
        
    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        if self.local_human:
            observation = {}
            observation['id'] = self.getID()
            reply_text = input("Enter Your Message: ")
            reply_text = self.preprocess(reply_text)
            observation['episode_done'] = True  ### TODO: for history
            
            """
            if '[DONE]' in reply_text:
                reply['episode_done'] = True
                self.episodeDone = True
                reply_text = reply_text.replace('[DONE]', '')
            """
            observation['text'] = reply_text
     
        else:
            # shallow copy observation (deep copy can be expensive)
            observation = observation.copy()
            if not self.episode_done and self.episode_concat: 
                # if the last example wasn't the end of an episode, then we need to
                # recall what was said in that example
                prev_dialogue = self.observation['text']
                observation['text'] = prev_dialogue + '\n' + observation['text'] #### TODO!!!! # DATA is concatenated!!
  
        self.observation = observation
        self.episode_done = observation['episode_done']          
        
        return observation

    def _encode(self, xs, xlen, dropout=False, packed=True):
        """Call encoder and return output and hidden states."""
        batchsize = len(xs)

        # first encode context
        xes = self.lt(xs).transpose(0, 1)
        #if dropout:
        #    xes = self.dropout(xes)
        
        # initial hidden 
        if self.zeros.size(1) != batchsize:
            if self.opt['bi_encoder']:   
                self.zeros.resize_(2*self.num_layers, batchsize, self.hidden_size).fill_(0) 
            else:
                self.zeros.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0) 
            
        h0 = Variable(self.zeros.fill_(0))
        
        # forward
        if packed:
            xes = torch.nn.utils.rnn.pack_padded_sequence(xes, xlen)
                
        if type(self.encoder) == nn.LSTM:
            encoder_output, _ = self.encoder(xes, (h0, h0)) ## Note : we can put None instead of (h0, h0)
        else:
            encoder_output, _ = self.encoder(xes, h0)
        
        if packed:
            encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_output)
            
        encoder_output = encoder_output.transpose(0, 1) #batch first
        
        """
        if self.use_attention:
            if encoder_output.size(1) > self.max_length:
                offset = encoder_output.size(1) - self.max_length
                encoder_output = encoder_output.narrow(1, offset, self.max_length)
        """
                
        return encoder_output


    def _apply_attention(self, word_input, encoder_output, last_hidden, xs):
        """Apply attention to encoder hidden layer."""
        batch_size = encoder_output.size(0)
        enc_length = encoder_output.size(1)
        mask = Variable(xs.data.eq(0).eq(0).float())
        
        #pdb.set_trace()
        # encoder_output # B x T x 2H
        # last_hidden  B x H

        if self.att_type == 'concat':
            last_hidden = last_hidden.unsqueeze(1).expand(batch_size, encoder_output.size(1), self.hidden_size) # B x T x H
            attn_weights = F.tanh(self.attn(torch.cat((encoder_output, last_hidden), 2).view(batch_size*enc_length,-1)).view(batch_size, enc_length))
        elif self.att_type == 'dot':
            attn_weights = F.tanh(torch.bmm(encoder_output, last_hidden.unsqueeze(2)).squeeze())
        elif self.att_type == 'general':
            attn_weights = F.tanh(torch.bmm(encoder_output, self.attn(last_hidden).unsqueeze(2)).squeeze())
            
        #attn_weights = F.softmax(attn_weights.view(batch_size, enc_length))

        attn_weights = attn_weights.exp().mul(mask)
        denom = attn_weights.sum(1).unsqueeze(1).expand_as(attn_weights)
        attn_weights = attn_weights.div(denom)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)
        
        output = torch.cat((word_input, context.unsqueeze(0)),2)
        return output

    def _get_context(self, batchsize, xlen_t, encoder_output):
        " return initial hidden of decoder and encoder context (last_state)"
        
        ## The initial of decoder is the hidden (last states) of encoder --> put zero!       
        if self.zeros_dec.size(1) != batchsize:
            self.zeros_dec.resize_(self.num_layers, batchsize, self.hidden_size).fill_(0)
        hidden = Variable(self.zeros_dec.fill_(0))
        
        last_state = None
        if not self.use_attention:
            last_state = torch.gather(encoder_output, 1, xlen_t.view(-1,1,1).expand(encoder_output.size(0),1,encoder_output.size(2)))
            if self.opt['bi_encoder']:
#                last_state = torch.cat((encoder_output[:,0,:self.hidden_size], last_state[:,0,self.hidden_size:]),1)        
                last_state = torch.cat((encoder_output[:,0,self.hidden_size:], last_state[:,0,:self.hidden_size]),1)        
        
        return hidden, last_state
    
    def _decode_and_train(self, batchsize, dec_xes, xlen_t, xs, ys, ylen, encoder_output):
        # update the model based on the labels
        self.zero_grad()
        loss = 0
        
        output_lines = [[] for _ in range(batchsize)]
        
        # keep track of longest label we've ever seen
        self.longest_label = max(self.longest_label, ys.size(1))
        
        hidden, last_state = self._get_context(batchsize, xlen_t, encoder_output)

        for i in range(ys.size(1)):
            if self.use_attention:
                output = self._apply_attention(dec_xes, encoder_output, hidden[-1], xs)
            else:                
                output = torch.cat((dec_xes, last_state.unsqueeze(0)), 2)
            
            output, hidden = self.decoder(output, hidden)           
            preds, scores = self.hidden_to_idx(output, dropout=self.training)
            y = ys.select(1, i)
            loss += self.criterion(scores, y) #not averaged
            # use the true token as the next input instead of predicted
            # this produces a biased prediction but better training
            dec_xes = self.lt(y).unsqueeze(0)
            
            # TODO: overhead!
            for b in range(batchsize):
                # convert the output scores to tokens
                token = self.v2t([preds.data[b]])
                output_lines[b].append(token)
        
        if self.training:
            self.loss = loss.data[0]/sum(ylen) # consider non-NULL
            self.ndata += batchsize
        else:
            self.loss_valid += loss.data[0] # consider non-NULL / accumulate!
            self.ndata_valid += sum(ylen)          

        return loss, output_lines

    def _decode_only(self, batchsize, dec_xes, xlen_t, xs, encoder_output):
        # just produce a prediction without training the model
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0

        output_lines = [[] for _ in range(batchsize)]

        hidden, last_state = self._get_context(batchsize, xlen_t, encoder_output)

        # now, generate a response from scratch
        while(total_done < batchsize) and max_len < self.longest_label:
            # keep producing tokens until we hit END or max length for each
            if self.use_attention:
                output = self._apply_attention(dec_xes, encoder_output, hidden[-1], xs)
            else:                
                output = torch.cat((dec_xes, last_state.unsqueeze(0)), 2)            
            
            output, hidden = self.decoder(output, hidden)
            preds, scores = self.hidden_to_idx(output, dropout=False)

            #dec_xes = self.lt2dec(self.lt(preds.unsqueeze(0)))
            dec_xes = self.lt(preds).unsqueeze(0)
            
            max_len += 1
            for b in range(batchsize):
                if not done[b]:
                    # only add more tokens for examples that aren't done yet
                    token = self.v2t([preds.data[b]])
                    if token == self.END:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
                    else:
                        output_lines[b].append(token)

        return output_lines

    def _beam_search(self, batchsize, dec_xes, xlen_t, xs, encoder_output, n_best=20):
        # Code borrowed from PyTorch OpenNMT example
        # https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/decode.py
        
        print('(beam search {})'.format(self.beamsize))
        
        # just produce a prediction without training the model
        done = [False for _ in range(batchsize)]
        total_done = 0
        max_len = 0
        output_lines = [[] for _ in range(batchsize)]
                
        hidden, last_state = self._get_context(batchsize, xlen_t, encoder_output)  ## hidden = 2(#layer)x1x2048 / last_state = 1x4096
                
        # exapnd tensors for each beam
        beamsize = self.beamsize
        if not self.use_attention:
            context = Variable(last_state.data.repeat(1, beamsize, 1))
            
        dec_states = [
            Variable(hidden.data.repeat(1, beamsize, 1)) # 2x3x2048
            #Variable(context_c_t.data.repeat(1, self.beamsize, 1)) ## TODO : GRU OK. check LSTM ?
        ]

        beam = [ Beam(beamsize, self.dict.tok2ind, cuda = self.use_cuda) for k in range(batchsize) ]        
        
        batch_idx = list(range(batchsize))
        remaining_sents = batchsize
        
        input = Variable(dec_xes.data.repeat(1, beamsize, 1))
        encoder_output = Variable(encoder_output.data.repeat(beamsize,1, 1))
        
        while max_len < self.max_seq_len:
            
            # keep producing tokens until we hit END or max length for each
            if self.use_attention: 
                output = self._apply_attention(input, encoder_output, dec_states[0][-1], xs)
            else:                
                output = torch.cat((input, context), 2)         
            
            output, hidden = self.decoder(output, dec_states[0]) 
            preds, scores = self.hidden_to_idx(output, dropout=False)
            
            dec_states = [hidden]
            word_lk = scores.view(beamsize, remaining_sents, -1).transpose(0, 1).contiguous()     
            
            active = []
            for b in range(batchsize):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                #if not beam[b].advance(word_lk.data[idx]):
                #if not beam[b].advance_end(word_lk.data[idx]):
                if not beam[b].advance_diverse(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(-1, beamsize, remaining_sents, dec_state.size(2))[:, :, idx]
                    sent_states.data.copy_(sent_states.data.index_select( 1, beam[b].get_current_origin()))

            if not active:
                break
            
            """
            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remaining_sents, self.decoder.hidden_size)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select( 1, active_idx).view(*new_size))

            pdb.set_trace()
            dec_states = (
                update_active(dec_states[0])#, 2x3x2048  #layer x batch*beam * 2048
                #update_active(dec_states[1])
            )
            #dec_out = update_active(dec_out)
            context = update_active(context) # 1x3x4096

            remaining_sents = len(active)
            pdb.set_trace()            
            """
            
            input = torch.stack([b.get_current_state() for b in beam if not b.done]).t().contiguous().view(1, -1)
            input = self.lt(Variable(input))    

            max_len += 1
            
        all_preds, allScores = [], []
        for b in range(batchsize):        ## TODO :: does it provide batchsize > 1 ?
            hyps = []
            scores, ks = beam[b].sort_best()

            allScores += [scores[:self.beamsize]]
            hyps += [beam[b].get_hyp(k) for k in ks[:self.beamsize]]
            
            all_preds += [' '.join([self.dict.ind2tok[y] for y in x if not y is 0]) for x in hyps] # self.dict.null_token = 0
            
            if n_best == 1:
                print('\n    input:', self.dict.vec2txt(xs[0].data.cpu()).replace(self.dict.null_token+' ', ''),
                  '\n    pred :', ''.join(all_preds[b]), '\n')
            else:
                print('\n    input:', self.dict.vec2txt(xs[0].data.cpu()).replace(self.dict.null_token+' ', '\n'))
                for hyps in range(len(hyps)):
                    print('   {:3f} '.format(scores[hyps]), ''.join(all_preds[hyps]))

            print('the first: '+ ' '.join([self.dict.ind2tok[y] for y in beam[0].nextYs[1]]))
        return [all_preds[0]] # 1-best

    def _score_candidates(self, cands, xe, encoder_output, hidden):
        # score each candidate separately

        # cands are exs_with_cands x cands_per_ex x words_per_cand
        # cview is total_cands x words_per_cand
        cview = cands.view(-1, cands.size(2))
        cands_xes = xe.expand(xe.size(0), cview.size(0), xe.size(2))
        sz = hidden.size()
        cands_hn = (
            hidden.view(sz[0], sz[1], 1, sz[2])
            .expand(sz[0], sz[1], cands.size(1), sz[2])
            .contiguous()
            .view(sz[0], -1, sz[2])
        )

        sz = encoder_output.size()
        cands_encoder_output = (
            encoder_output.contiguous()
            .view(sz[0], 1, sz[1], sz[2])
            .expand(sz[0], cands.size(1), sz[1], sz[2])
            .contiguous()
            .view(-1, sz[1], sz[2])
        )

        cand_scores = Variable(
                    self.cand_scores.resize_(cview.size(0)).fill_(0))
        cand_lengths = Variable(
                    self.cand_lengths.resize_(cview.size(0)).fill_(0))

        for i in range(cview.size(1)):
            output = self._apply_attention(cands_xes, cands_encoder_output, cands_hn) \
                    if self.use_attention else cands_xes

            output, cands_hn = self.decoder(output, cands_hn)
            preds, scores = self.hidden_to_idx(output, dropout=False)
            cs = cview.select(1, i)
            non_nulls = cs.ne(self.NULL_IDX)
            cand_lengths += non_nulls.long()
            score_per_cand = torch.gather(scores, 1, cs.unsqueeze(1))
            cand_scores += score_per_cand.squeeze() * non_nulls.float()
            #cands_xes = self.lt2dec(self.lt(cs).unsqueeze(0))
            cands_xes = self.lt(cs).unsqueeze(0)

        # set empty scores to -1, so when divided by 0 they become -inf
        cand_scores -= cand_lengths.eq(0).float()
        # average the scores per token
        cand_scores /= cand_lengths.float()

        cand_scores = cand_scores.view(cands.size(0), cands.size(1))
        srtd_scores, text_cand_inds = cand_scores.sort(1, True)
        text_cand_inds = text_cand_inds.data

        return text_cand_inds

    def predict(self, xs, xlen, ylen=None, ys=None, cands=None):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available.
        """
        
        self._training(self.training)
        
        batchsize = len(xs)
        text_cand_inds = None
        target_exist = ys is not None
        
        xlen_t = torch.LongTensor(xlen)-1
        if self.use_cuda:
            xlen_t = xlen_t.cuda()
        xlen_t = Variable(xlen_t)
                
        # Encoding 
        encoder_output = self._encode(xs, xlen, dropout=self.training)

        # next we use START as an input to kick off our decoder
        x = Variable(self.START_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        dec_xes = xe.expand(xe.size(0), batchsize, xe.size(2))

        # list of output tokens for each example in the batch
        output_lines = None
        
        # Decoding        
        if not self.generating:
        #if (target_exist is not None) and (self.generating is False):
            loss, output_lines = self._decode_and_train(batchsize, dec_xes, xlen_t, xs, ys, ylen,
                                                  encoder_output)
            if self.training:
                loss.backward()
                if self.opt['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm(self.lt.parameters(), self.opt['grad_clip'])
                    torch.nn.utils.clip_grad_norm(self.h2o.parameters(), self.opt['grad_clip'])                    
                    torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.opt['grad_clip'])
                    torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.opt['grad_clip'])
                self.update_params()
            self.display_predict(xs, ys, output_lines)

        else:
        #elif not target_exists or self.generating:
            assert(not self.training)
            if cands is not None:
                text_cand_inds = self._score_candidates(cands, xe, encoder_output)

            if self.opt['beam_size'] > 0:
                output_lines = self._beam_search(batchsize, dec_xes, xlen_t, xs, encoder_output)
            else:
                output_lines = self._decode_only(batchsize, dec_xes, xlen_t, xs, encoder_output)
                self.display_predict(xs, ys, output_lines, 1)
                    
        return output_lines, text_cand_inds

    def display_predict(self, xs, ys, output_lines, freq=0.01):
        if random.random() < freq:
            # sometimes output a prediction for debugging
            print('\n    input:', self.dict.vec2txt(xs[0].data.cpu()).replace(self.dict.null_token+' ', ''),
                  '\n    pred :', ' '.join(output_lines[0]), '\n')
            if ys is not None:
                print('    label:', self.dict.vec2txt(ys[0].data.cpu()).replace(self.dict.null_token+' ', ''), '\n')

    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]
        # the indices of the valid (non-empty) tensors
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        # set up the input tensors
        batchsize = len(exs)
        # tokenize the text
        xs = None
        xlen = None
        if batchsize > 0:
            parsed = [self.dict.parse(self.START)+self.parse(ex['text'])+self.dict.parse(self.END) for ex in exs]
            max_x_len = max([len(x) for x in parsed])            
            if self.truncate:
                # shrink xs to to limit batch computation
                max_x_len = min(max_x_len, self.max_seq_len)
                parsed = [x[-max_x_len:] for x in parsed]
        
            # sorting for unpack in encoder
            parsed_x = sorted(parsed, key=lambda p: len(p), reverse=True)            
            xlen = [len(x) for x in parsed_x]            
            xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
            
            """
            # pack the data to the right side of the tensor for this model
            for i, x in enumerate(parsed):
                offset = max_x_len - len(x)
                for j, idx in enumerate(x):
                    xs[i][j + offset] = idx
                    """
            for i, x in enumerate(parsed_x):
                for j, idx in enumerate(x):
                    xs[i][j] = idx        
            if self.use_cuda:
                # copy to gpu
                self.xs.resize_(xs.size())
                self.xs.copy_(xs, async=True)
                xs = Variable(self.xs)
            else:
                xs = Variable(xs)
            
        # set up the target tensors
        ys = None
        ylen = None
        
        if batchsize > 0 and (any(['labels' in ex for ex in exs]) or any(['eval_labels' in ex for ex in exs])):
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            if any(['labels' in ex for ex in exs]):
                labels = [random.choice(ex.get('labels', [''])) + ' ' + self.END for ex in exs]
            else:
                labels = [random.choice(ex.get('eval_labels', [''])) + ' ' + self.END for ex in exs]
                
            parsed_y = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed_y)
            if self.truncate:
                # shrink ys to to limit batch computation
                max_y_len = min(max_y_len, self.max_seq_len)
                parsed_y = [y[:max_y_len] for y in parsed_y]
            
            seq_pairs = sorted(zip(parsed, parsed_y), key=lambda p: len(p[0]), reverse=True)
            _, parsed_y = zip(*seq_pairs)
                            
            ylen = [len(x) for x in parsed_y]
            ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(parsed_y):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                # copy to gpu
                self.ys.resize_(ys.size())
                self.ys.copy_(ys, async=True)
                ys = Variable(self.ys)
            else:
                ys = Variable(ys)
                
        # set up candidates
        cands = None
        valid_cands = None
        if ys is None and self.rank:
            # only do ranking when no targets available and ranking flag set
            parsed = []
            valid_cands = []
            for i in valid_inds:
                if 'label_candidates' in observations[i]:
                    # each candidate tuple is a pair of the parsed version and
                    # the original full string
                    cs = list(observations[i]['label_candidates'])
                    parsed.append([self.parse(c) for c in cs])
                    valid_cands.append((i, cs))
            if len(parsed) > 0:
                # TODO: store lengths of cands separately, so don't have zero
                # padding for varying number of cands per example
                # found cands, pack them into tensor
                max_c_len = max(max(len(c) for c in cs) for cs in parsed)
                max_c_cnt = max(len(cs) for cs in parsed)
                cands = torch.LongTensor(len(parsed), max_c_cnt, max_c_len).fill_(0)
                for i, cs in enumerate(parsed):
                    for j, c in enumerate(cs):
                        for k, idx in enumerate(c):
                            cands[i][j][k] = idx
                if self.use_cuda:
                    # copy to gpu
                    self.cands.resize_(cands.size())
                    self.cands.copy_(cands, async=True)
                    cands = Variable(self.cands)
                else:
                    cands = Variable(cands)

        return xs, ys, valid_inds, cands, valid_cands, xlen, ylen

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, valid_inds, cands, valid_cands, xlen, ylen = self.batchify(observations)

        if xs is None:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        
        predictions, text_cand_inds = self.predict(xs, xlen, ylen, ys, cands)
        #pdb.set_trace()
        
        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[valid_inds[i]]
            #curr['text'] = ' '.join(c for c in predictions[i] if c != self.END and c != self.dict.null_token) ## TODO: check!!
            curr['text'] = ''.join(c for c in predictions[i] if c != self.END and c != self.dict.null_token) ## TODO: check!!
            

        if text_cand_inds is not None:
            for i in range(len(valid_cands)):
                order = text_cand_inds[i]
                batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[batch_idx]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]
        
        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'lt'):
            model = {}
            model['lt'] = self.lt.state_dict()
            #model['lt2enc'] = self.lt2enc.state_dict()
            #model['lt2dec'] = self.lt2dec.state_dict()
            model['encoder'] = self.encoder.state_dict()
            model['decoder'] = self.decoder.state_dict()
            model['h2o'] = self.h2o.state_dict()
            if self.use_attention:
                model['attn'] = self.attn.state_dict()
            model['optims'] = {k: v.state_dict()
                               for k, v in self.optims.items()}
            model['longest_label'] = self.longest_label
            model['opt'] = self.opt

            with open(path, 'wb') as write:
                torch.save(model, write)

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def load(self, path):
        """Return opt and model states."""
        with open(path, 'rb') as read:
            if(self.use_cuda):
                model = torch.load(read)
            else:
                model = torch.load(read, map_location=lambda storage, loc: storage)
        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.lt.load_state_dict(states['lt'])
        self.encoder.load_state_dict(states['encoder'])
        self.decoder.load_state_dict(states['decoder'])
        self.h2o.load_state_dict(states['h2o'])
        if self.use_attention:
            self.attn.load_state_dict(states['attn'])
        for k, v in states['optims'].items():
            self.optims[k].load_state_dict(v)
        self.longest_label = states['longest_label']

    def report(self):
        m={}
        if not self.generating:
            if self.training:
                m['nll'] = self.loss
                m['ppl'] = math.exp(self.loss)
                m['ndata'] = self.ndata
            else:
                m['nll'] = self.loss_valid/self.ndata_valid
                m['ppl'] = math.exp(self.loss_valid/self.ndata_valid)
                m['ndata'] = self.ndata_valid
                            
            m['lr'] = self.lr
            self.print_weight_state()
        
        return m
    
    def reset_valid_report(self):
        self.ndata_valid = 0
        self.loss_valid = 0
        
        
    def print_weight_state(self):
        self._print_grad_weight(getattr(self, 'lt').weight, 'lookup')
        for module in {'encoder', 'decoder'}:
            layer = getattr(self, module)
            for weights in layer._all_weights:  
                for weight_name in weights:
                    self._print_grad_weight(getattr(layer, weight_name), module + ' '+ weight_name)
        self._print_grad_weight(getattr(self, 'h2o').weight, 'h2o')
        if self.use_attention:
           self._print_grad_weight(getattr(self, 'attn').weight, 'attn')
                
    def _print_grad_weight(self, weight, module_name):
        if weight.dim() == 2:
            nparam=weight.size(0) * weight.size(1)
            norm_w = weight.norm(2).pow(2)
            norm_dw = weight.grad.norm(2).pow(2)
            print('{:30}'.format(module_name) + ' {:5} x{:5}'.format(weight.size(0), weight.size(1))
                   + ' : w {0:.2e} | '.format((norm_w/nparam).sqrt().data[0]) + 'dw {0:.2e}'.format((norm_dw/nparam).sqrt().data[0]))

    def _training(self, training=True):
        for module in {'encoder', 'decoder', 'lt', 'h2o', 'attn'}:
            layer = getattr(self, module)
            if layer is not None:
                layer.training=training
            
            
    
