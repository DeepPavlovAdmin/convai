# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Script to run evaluation only on a pretrained model."""
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')
import logging

import pdb, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy

from parlai.agents.drqa.utils import Timer, normalize_text
from parlai.agents.drqa.agents import DocReaderAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

try:
    import spacy
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install spacy and spacy 'en' model: go to spacy.io"
    )
NLP = spacy.load('en')

def subplot(matrix, plt, fig, nrow, ncol, n, yticklabel, xticklabel, title):
    ax1 = fig.add_subplot(nrow, ncol, n)
    ax1.matshow(matrix.data.cpu().numpy())
    ax1.set_xticks(range(len(xticklabel)))
    ax1.set_xticklabels(xticklabel, rotation = 'vertical')
    ax1.set_yticks(range(len(yticklabel)))
    ax1.set_yticklabels(yticklabel)
    ax1.set_title(title)
    plt.tight_layout()
    
def main(opt):
    # Check options
    assert('pretrained_model' in opt)
    assert(opt['datatype'] in {'test', 'valid'})

    # Calculate TDNN embedding dim (after applying TDNN to char tensor)
    opt['kernels'] = ''.join(opt['kernels'])
    if isinstance(opt['kernels'], str):
        opt['kernels'] = eval(opt['kernels']) # convert string list of tuple --> list of tuple

    if opt['add_char2word']:
        opt['NULLWORD_Idx_in_char'] = opt['vocab_size_char']-1
        opt['embedding_dim_TDNN']=0
        for i, n in enumerate(opt['kernels']):
            opt['embedding_dim_TDNN'] += n[1]

        logger.info('TDNN embedding dim = %d' % (opt['embedding_dim_TDNN']))

    # Load document reader
    doc_reader = DocReaderAgent(opt)
    
    logger.info('[ Running validation... ]')
    valid_world = create_task(opt, doc_reader)
    valid_time = Timer()
      
    nExample = 0
    for _ in valid_world:
        valid_world.parley()
        nExample+=1
        
        if nExample % 10 == 0:
            
            text = valid_world.acts[0]['text']        
            p = NLP.tokenizer(text.split('\n')[0])
            q = NLP.tokenizer(text.split('\n')[1])
                    
            ## ATTENTION weights
            qemb = valid_world.agents[1].model.network.qemb_match.get_alpha().squeeze()       
            qp_attention = valid_world.agents[1].model.network.qp_match.get_alpha().squeeze()
            #qp_gate = valid_world.agents[1].model.network.qp_match.get_gate().squeeze()
    
            pp_attention = valid_world.agents[1].model.network.pp_match.get_alpha().squeeze()
    #        pp_gate = valid_world.agents[1].model.network.pp_match.get_gate().squeeze()
            
            s_att = valid_world.agents[1].model.network.start_attn.get_alpha()
            e_att = valid_world.agents[1].model.network.end_attn.get_alpha()
            
            q_merge = valid_world.agents[1].model.network.self_attn.get_alpha()
            
            #pdb.set_trace()
            fig = plt.figure(1, figsize=(30,30))
            i=4        
            subplot(qemb, plt, fig, 1, i, 1, p, q, 'q_emb')
            subplot(qp_attention, plt, fig, 1, i, 2, p, q, 'qp-att')
            #subplot(torch.cat([qp_gate, pp_gate],0).t(), plt, fig, 1, 5, 3, p, ['qp-gate', 'pp-gate'], '')
            #subplot(pp_gate.t(), plt, fig, 1, 5, 4, p, [''], 'pp-gate')
            
            subplot(torch.cat([s_att, e_att],0).t(), plt, fig, 1, i, 3, p, ['start', 'end'], valid_world.agents[0].lastY_prev[0])
            subplot(q_merge.t(), plt, fig, 1, i, 4, q, [' '], 'q-merging')

            
            fig.savefig((str(opt['model_file']) + '_'+ str(nExample) +'.png'), transparent=True)
            plt.close()
            
            fig = plt.figure(2, figsize=(30,30))        
            subplot(pp_attention, plt, fig, 1, 1, 1, p, p, 'pp-att')        
            fig.savefig((str(opt['model_file']) + '_'+ str(nExample) +'pp.png'), transparent=True)
            plt.close()
        
        """
        subplot(qemb.t(), plt, fig, 4, 1, 1, q, p)
        subplot(qp_attention.t(), plt, fig, 4, 1, 2, q, p
        subplot(pp_attention.t(), plt, fig, 4, 1, 3, p, p)
        subplot(torch.cat([s_att, e_att],0), plt, fig, 4, 4, ['start', 'end'], p)
        """      
        
        if nExample == 100:
            break
        
   
if __name__ == '__main__':
    # Get command line arguments
    argparser = ParlaiParser()
    DocReaderAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    # Set logging (only stderr)
    logger = logging.getLogger('DrQA')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Set cuda
    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        logger.info('[ Using CUDA (GPU %d) ]' % opt['gpu'])
        torch.cuda.set_device(opt['gpu'])

    # Run!
    main(opt)
