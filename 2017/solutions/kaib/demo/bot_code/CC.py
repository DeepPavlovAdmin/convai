try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')

import sys
sys.path.append("../ParlAI")

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task, validate
from parlai.core.params import ParlaiParser

import random
import logging, sys, os

import pdb

class CC:
    def __init__(self, pretrained_mdl_path, dict_dir, cuda=False):
        print('initialize CC module')
        
        def set_opt(opt):
            opt['task'] = 'dailydialog' # opensubtitles
            opt['beam_size'] = 30
            opt['max_seq_len'] = 50
            opt['dict_file'] = dict_dir
            opt['gpu'] = -1 # use cpu
            opt['cuda'] = cuda
            opt['no_cuda'] = (not cuda)
            opt['batchsize'] = 1
            opt['weight_decay'] = 0
            if 'add_char2word' not in opt:
                opt['add_char2word'] = False
            return opt
        
        
        ## load seq2sq_v2
        opt = get_opt(pretrained_mdl_path, cuda)
        opt['model_file'] = pretrained_mdl_path        
        opt['model'] = 'seq2seq_v2'
        opt = set_opt(opt)                
        self.opt = opt
        self.agent = create_agent(opt)        
        self.agent.training= False
        self.agent.generating = True
                
        # self.world = create_task(opt, self.agent)
        

    def get_reply(self, message, history_context=None, history_reply=None):

        observation = {}
        message = self.agent.preprocess(message)
        observation['episode_done'] = True  ### TODO: for history
        observation['text'] = message
    
        self.agent.observe(validate(observation))
        response = self.agent.act()        
        response = self.post_proceessing(response['text'])
    
        return response
    
    def post_proceessing(self, response):
        ## Valid answer should be selected!!
        
        response = response.replace(self.agent.dict.null_token, "")
        response = response.replace(self.agent.dict.unk_token, "**") ## TODO : response including unk should not be selected
        response = response.replace(self.agent.dict.end_token, "")
        
        reply_text=response.replace(" 'm", "'m")
        reply_text=reply_text.replace(" 've", "'ve")
        reply_text=reply_text.replace(" 's", "'s")
        reply_text=reply_text.replace(" 't", "'t")
        reply_text=reply_text.replace(" 'il", "'il")
        reply_text=reply_text.replace(" 'd", "'d")
        reply_text=reply_text.replace(" 're", "'re")        
        reply_text=reply_text.replace(" ?", "?")        
        reply_text=reply_text.replace(" .", ".")        
        reply_text=reply_text.replace(" !", "!")        
        reply_text=reply_text.strip()

        #reply_text[1].upper()
        return reply_text        
    
def get_opt(pretrained_model_path, cuda=False):
    print("load model: ", pretrained_model_path)
    if cuda: 
        print('(CC) load model opt from gpu')
        mdl = torch.load(pretrained_model_path)
    else:
        print('(CC) load model opt from cpu')
        mdl = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        
    opt = mdl['opt']
    del mdl

    return opt
    
if __name__ == "__main__":
    
    root_dir = '../ParlAI-v2'
    exp_dir = 'exp-dailydialog' #'exp-opensub'
    
    #model_name = 'exp-emb300-hs1024-lr0.0001-gc0.5'
    model_name = 'exp-emb300-hs2048-lr0.0001-bs128'

    pretrained_mdl_path = os.path.join(root_dir, exp_dir, model_name, model_name)  # release ver
    
    dict_dir = os.path.join(root_dir, exp_dir, 'dict_file_th5.dict')
        
    cc = CC(pretrained_mdl_path, dict_dir, cuda=False)

    # Example1 (in train)
    question_sample = "How many BS level degrees are offered in the College of Engineering at Notre Dame?"

    print(cc.get_reply(question_sample))
    print(cc.get_reply('How are you ?'))
    print(cc.get_reply('and it participated in the olympics as a sponsor ?'))
    print(cc.get_reply('i lovv you'))
    #print(cc.get_reply('welcome!!!!'))
    pdb.set_trace()
    
    
