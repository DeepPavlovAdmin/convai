# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Hwaran Lee, KAIST: 2017-present

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch

import os
import random, math
import pdb

class ScoringNetAgent(Agent):
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
                           choices=ScoringNetAgent.ENC_OPTS.keys(),
                           help='Choose between different encoder modules.')
        agent.add_argument('-bi', '--bi-encoder', default=True, type='bool',
                           help='Bidirection of encoder')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'] + list(ScoringNetAgent.ENC_OPTS.keys()),
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights.')
        agent.add_argument('-opt', '--optimizer', default='sgd',
                           choices=ScoringNetAgent.OPTIM_OPTS.keys(),
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
        agent.add_argument('-ptrmodel', '--ptr_model', default='',
                           help='The pretrained model directory')
        
                
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
            
            """
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, self.states = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.override_opt(new_opt)
            """
            if opt.get('ptr_model') and os.path.isfile(opt['ptr_model']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['ptr_model'])
                new_opt, self.states = self.load(opt['ptr_model']) ## TODO:: load what?
                # override options with stored ones
                #opt = self.override_opt(new_opt)
                 
            self.dict = DictionaryAgent(opt)
            self.id = 'ScoringNet'
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
            self.neg_ys = torch.LongTensor(1, 1)
            

            # set up modules
            #self.criterion = nn.NLLLoss(size_average = False, ignore_index = 0)
            self.criterion = nn.BCELoss()
            
            # lookup table stores word embeddings
            self.lt = nn.Embedding(len(self.dict), emb,
                                   padding_idx=self.NULL_IDX)
                                   #scale_grad_by_freq=True)
            # encoder captures the input text
            enc_class = ScoringNetAgent.ENC_OPTS[opt['encoder']]
            self.encoder = enc_class(emb, hsz, opt['numlayers'], bidirectional=opt['bi_encoder'], dropout = opt['dropout'])
            # decoder produces our output states
            
            dec_isz = hsz
            if opt['bi_encoder']:
                dec_isz += hsz
            
            # linear layer helps us produce outputs from final decoder state
            self.h2o = nn.Linear(dec_isz, dec_isz, bias=False)
            
            
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

            # set up optims for each module
            self.lr = opt['learning_rate']
            self.wd = opt['weight_decay'] is not 0

            optim_class = ScoringNetAgent.OPTIM_OPTS[opt['optimizer']]
            self.optims = {
                'lt': optim_class(self.lt.parameters(), lr=self.lr),
                'encoder': optim_class(self.encoder.parameters(), lr=self.lr),
                'h2o': optim_class(self.h2o.parameters(), lr=self.lr, weight_decay=self.wd),                
            }
            if self.attention and self.attn is not None:
                self.optims.update({'attn': optim_class(self.attn.parameters(), lr=self.lr, weight_decay=self.wd)})
            
            if hasattr(self, 'states'):
                # set loaded states if applicable
                if opt.get('ptr_model'):
                    self.init_pretrain(self.states)
                else:
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
        self.max_seq_len = opt['max_seq_len']
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
                      'encoder'}
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
        self.neg_ys = self.neg_ys.cuda(async=True)
        self.criterion.cuda()
        self.lt.cuda()
        self.encoder.cuda()
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
                last_state = torch.cat((encoder_output[:,0,self.hidden_size:], last_state[:,0,:self.hidden_size]),1)        
        
        return hidden, last_state
    
    def predict(self,xs, xlen, x_idx, ys, ylen, y_idx, nys=None, nylen=None, ny_idx=None):
        
        
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available.
        """
        
        self._training(self.training)
        self.zero_grad()

        batchsize = len(xs)
        #text_cand_inds = None
        #target_exist = ys is not None        
        
        xlen_t = Variable(torch.LongTensor(xlen)-1)
        ylen_t = Variable(torch.LongTensor(ylen)-1)
        if self.use_cuda:
            xlen_t = xlen_t.cuda()            
            ylen_t = ylen_t.cuda()            
        
        _, x_idx_t = torch.LongTensor(x_idx).sort(0)
        _, y_idx_t = torch.LongTensor(y_idx).sort(0)       
                       
        if self.use_cuda:
            x_idx_t = x_idx_t.cuda()
            y_idx_t = y_idx_t.cuda()
        if ny_idx is not None:
            nylen_t = Variable(torch.LongTensor(nylen)-1)
            _, ny_idx_t = torch.LongTensor(ny_idx).sort(0)
            
            if self.use_cuda:
                nylen_t = nylen_t.cuda()
                ny_idx_t = ny_idx_t.cuda()
               
        # Encoding 
        _, enc_x = self._get_context(batchsize, xlen_t, self._encode(xs, xlen, dropout=self.training))   # encode x
        _, enc_y = self._get_context(batchsize, ylen_t, self._encode(ys, ylen, dropout=self.training))   # encode x
        _, enc_ny = self._get_context(batchsize, nylen_t, self._encode(nys, nylen, dropout=self.training))   # encode x

        # Permute
        enc_x = enc_x[x_idx_t, :]
        enc_y = enc_y[y_idx_t, :]
        enc_ny= enc_ny[ny_idx_t, :]
        
        # make batch
        enc_x = torch.cat((enc_x, enc_x), 0)
        enc_y = torch.cat((enc_y, enc_ny), 0)
        
        target = Variable(torch.Tensor(batchsize).zero_())        
        target = torch.cat((target, target+1), 0)        
        if self.use_cuda:
            target = target.cuda()
            
        # calcuate the score
        output = F.sigmoid(torch.bmm(enc_y.unsqueeze(1), self.h2o(enc_x).unsqueeze(1).transpose(1,2)))

        # loss
        loss = self.criterion(output.squeeze(), target) 
                           
        if self.training:
            self.ndata += batchsize
            self.loss = loss
        else:
            self.ndata_valid += batchsize
            self.loss_valid += loss.data[0]*batchsize
        
        # list of output tokens for each example in the batch
        if self.training:
            self.loss.backward()
            if self.opt['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm(self.lt.parameters(), self.opt['grad_clip'])
                torch.nn.utils.clip_grad_norm(self.h2o.parameters(), self.opt['grad_clip'])                    
                torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.opt['grad_clip'])
            self.update_params()
        
        self.display_predict(xs[x_idx_t[0], :], ys[y_idx_t[0], :], nys[ny_idx_t[0], :], target, output, batchsize, freq=0.05)
        
        return self.loss

    def display_predict(self, xs, ys, nys, target, output, batchsize, freq=0.01):
        if random.random() < freq:
            # sometimes output a prediction for debugging
            print('\n    input:', self.dict.vec2txt(xs.data.cpu()).replace(self.dict.null_token+' ', ''),
                  '\n    postive:', ' {0:.2e} '.format(output[0].data.cpu()[0,0]), self.dict.vec2txt(ys.data.cpu()).replace(self.dict.null_token+' ', ''),
                  '\n    negative:', ' {0:.2e} '.format(output[batchsize].data.cpu()[0,0]),self.dict.vec2txt(nys.data.cpu()).replace(self.dict.null_token+' ', ''), '\n')

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
        x_idx = None
        if batchsize > 0:
            parsed = [self.dict.parse(self.START)+self.parse(ex['text'])+self.dict.parse(self.END) for ex in exs]
            max_x_len = max([len(x) for x in parsed])            
            if self.truncate:
                # shrink xs to to limit batch computation
                max_x_len = min(max_x_len, self.max_seq_len)
                parsed = [x[-max_x_len:] for x in parsed]
        
            # sorting for unpack in encoder
            parsed_x = sorted(enumerate(parsed), key=lambda p: len(p[1]), reverse=True)
            x_idx, parsed_x = zip(*parsed_x)
            x_idx = list(x_idx)
            xlen = [len(x) for x in parsed_x]
            xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
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
            
        # set up the target tensors (positive exampels)
        ys = None
        ylen = None
        y_idx = None
                
        if batchsize > 0 and (any(['labels' in ex for ex in exs]) or any(['eval_labels' in ex for ex in exs])):
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            if any(['labels' in ex for ex in exs]):
                labels = [self.START + ' ' + random.choice(ex.get('labels', [''])) + ' ' + self.END for ex in exs]
            else:
                labels = [self.START + ' ' +random.choice(ex.get('eval_labels', [''])) + ' ' + self.END for ex in exs]

            parsed_y = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed_y)
            if self.truncate:
                # shrink ys to to limit batch computation
                max_y_len = min(max_y_len, self.max_seq_len)
                parsed_y = [y[:max_y_len] for y in parsed_y]
            
            # sorting for unpack in encoder
            parsed_y = sorted(enumerate(parsed_y), key=lambda p: len(p[1]), reverse=True)
            y_idx, parsed_y = zip(*parsed_y)
            y_idx = list(y_idx)                            
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
                
        # set up candidates (negative samples, randomly select!!)
        neg_ys = None
        neg_ylen = None        
        ny_idx = None        
      
        if batchsize > 0 :
            cands=None
            for i in range(len(exs)):
                if exs[i].get('label_candidates') is not None:
                    cands = list(exs[i]['label_candidates'])
                    break
            if cands is None:                
                if any(['labels' in ex for ex in exs]):
                    cands = [ ex['labels'][0] for ex in exs] ## TODO: the same index should not be selected
                else:
                    cands = [ ex['eval_labels'][0] for ex in exs] ## TODO: the same index should not be selected
            
            
            # randomly select one of the labels to update on, if multiple
            # append END to each label            
            parsed_ny = [self.dict.parse(self.START)+self.parse(random.choice(cands))+self.dict.parse(self.END) for ex in exs ]
            max_ny_len = max([len(x) for x in parsed_ny])            
            if self.truncate:
                # shrink xs to to limit batch computation
                max_ny_len = min(max_ny_len, self.max_seq_len)
                parsed_ny = [ny[-max_ny_len:] for ny in parsed_ny]
        
            # sorting for unpack in encoder
            parsed_ny = sorted(enumerate(parsed_ny), key=lambda p: len(p[1]), reverse=True)
            ny_idx, parsed_ny = zip(*parsed_ny)
            ny_idx = list(ny_idx)      
            
            neg_ylen = [len(x) for x in parsed_ny]
            neg_ys = torch.LongTensor(batchsize, max_ny_len).fill_(0)        
            for i, x in enumerate(parsed_ny):
                for j, idx in enumerate(x):
                    neg_ys[i][j] = idx        
            if self.use_cuda:
                # copy to gpu
                self.neg_ys.resize_(neg_ys.size())
                self.neg_ys.copy_(neg_ys, async=True)
                neg_ys = Variable(self.neg_ys)
            else:
                neg_ys = Variable(neg_ys)

        return xs, xlen, x_idx, ys, ylen, y_idx, valid_inds, neg_ys, neg_ylen, ny_idx

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, xlen, x_idx, ys, ylen, y_idx, valid_inds, neg_ys, neg_ylen, ny_idx = self.batchify(observations)
        
        if xs is None:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        
        ## seperate : test code / train code 
        loss = self.predict(xs, xlen, x_idx, ys, ylen, y_idx, neg_ys, neg_ylen, ny_idx)
        
        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'lt'):
            model = {}
            model['lt'] = self.lt.state_dict()
            model['encoder'] = self.encoder.state_dict()
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
            model = torch.load(read)

        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.lt.load_state_dict(states['lt'])
        self.encoder.load_state_dict(states['encoder'])
        #self.h2o.load_state_dict(states['h2o'])
        if self.use_attention:
            self.attn.load_state_dict(states['attn'])
        for k, v in states['optims'].items():
            self.optims[k].load_state_dict(v)
        self.longest_label = states['longest_label']
        
    def init_pretrain(self, states):
        """Set the state dicts of the modules from saved states."""
        self.lt.load_state_dict(states['lt'])
        self.encoder.load_state_dict(states['encoder'])
        #self.h2o.load_state_dict(states['h2o'])
        """
        if self.use_attention:
            self.attn.load_state_dict(states['attn'])
        for k, v in states['optims'].items():
            self.optims[k].load_state_dict(v)
        self.longest_label = states['longest_label']
        """
        
    def report(self):
        m={}
        if not self.generating:
            if self.training:
                m['loss'] = self.loss.data[0]
                m['ndata'] = self.ndata
            else:
                m['loss'] = self.loss_valid/self.ndata_valid
                m['ndata'] = self.ndata_valid
                            
            m['lr'] = self.lr
            self.print_weight_state()
        
        return m
    
    def reset_valid_report(self):
        self.ndata_valid = 0
        self.loss_valid = 0
        
        
    def print_weight_state(self):
        self._print_grad_weight(getattr(self, 'lt').weight, 'lookup')
        for module in {'encoder'}:
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
        for module in {'encoder', 'lt', 'h2o', 'attn'}:
            layer = getattr(self, module)
            if layer is not None:
                layer.training=training
            
            
    
