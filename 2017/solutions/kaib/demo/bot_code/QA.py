try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')
import random
import sys

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task, validate
from parlai.core.params import ParlaiParser

sys.path.append("../ParlAI/parlai/agents/pqmn") # bot.sh
sys.path.append("../ParlAI")
from pqmn import PqmnAgent

import pdb

#from parlai.agents.drqa_msmarco.agents import DocReaderAgent
#from parlai.agents.drqa import DocReaderAgent


class QA:
    def __init__(self, opt, cuda=False):
        print('initialize QA module (NOTE : NO PREPROCESS CODE FOR NOW. IMPORT IT FROM CC)')

        # Check options
        assert('pretrained_model' in opt)
        assert(opt['datatype'] == 'valid') # SQuAD only have valid data
        opt['batchsize']=1 #online mode
        opt['cuda'] = cuda
        opt['no_cuda'] = True

        # Load document reader
        #self.doc_reader = DocReaderAgent(opt)
        #self.doc_reader = DrqaAgent(opt)

        # Ver2
        """
        self.doc_reader = PqmnAgent(opt)
        self.doc_reader.model.network.eval()
        """

        # Ver3
        self.agent = create_agent(opt)
        self.agent.model.network.eval()

        # self.world = create_task(opt, self.agent)


    def get_reply(self, passage="", question=""):
        # ver1
        #reply = self.doc_reader.QA_single(passage, question)

        # ver2
        observation = {}
        observation['episode_done'] = True
        """
        observation['document'] = passage
        observation['question'] = question
        """
        observation['text'] = passage + '\n' + question


        #pdb.set_trace()
        self.agent.observe(validate(observation))
        reply = self.agent.act()['text']

        # post-processing
        #reply = reply.replace(" ", "")

        randnum = random.uniform(0,1)
        if(randnum < 0.33):
            reply = 'Maybe ' + reply + '. What do you think?'
        elif(randnum >=0.33 and randnum <0.66):
            reply = 'The answer is ' + reply
        else:
            reply = 'That would be ' + reply
        return reply

# Config
def get_opt(pretrained_model_path):
    mdl = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
    opt = mdl['config']
    del mdl

    return opt


if __name__ == "__main__":
    # Ver1
    #pretrained_mdl_path = '../../ParlAI/exp-squad/qp-pp-basic'  # qp-pp-basic

    # Ver2
    pretrained_mdl_path = '../../model/kaib_qa.mdl'  # release ver

    opt =  get_opt(pretrained_mdl_path)
    opt['pretrained_model'] = pretrained_mdl_path
    opt['datatype'] = 'valid'
    opt['batchsize']=1 #online mode
    opt['cuda'] = False
    opt['no_cuda'] = True

    #print('DISABLE GLOVE EMBEDDING FOR DEBUG --> we don''t need it')
    opt['embedding_file'] = '' # we don't need embedding file
    #opt['embedding_file'] = '../../ParlAI-v2/data/glove.840B.300d.txt' # WE DON"T NEED IT !

    # Temporary options?
    #opt['pp_gate'] = False  # for ver mismatch? temporary?
    #opt['pp_rnn'] = True    # for ver mismatch? temporary?

    #pdb.set_trace()
    qa = QA(opt)


    # Example1 (in train)
    passage_sample = "The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study \u2013 aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering \u2013 with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively."
    question_sample = "How many BS level degrees are offered in the College of Engineering at Notre Dame?"

    # Example2 (in train)
    #passage_sample = "The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study \u2013 aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering \u2013 with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively."
    #question_sample = "What is the oldest structure at Notre Dame?"

    #print('length of passage = ')
    #print(len(passage_sample))

    #pdb.set_trace()
    print(qa.get_reply(passage_sample, question_sample))
