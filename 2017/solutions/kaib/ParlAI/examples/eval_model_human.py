# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Hwaran Lee, KAIST: 2017-present

"""Basic example which iterates through the tasks specified and
evaluates the given model on them.

For example:
`python examples/eval_model.py -t "babi:Task1k:2" -m "repeat_label"`
or
`python examples/eval_model.py -t "#CornellMovie" -m "ir_baseline" -mp "-lp 0.5"`
"""

import torch

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser

import random
import pdb
import logging, sys

from examples.train_model_seq2seq_ldecay import run_eval

def main():
    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.set_defaults(datatype='valid')
    parser.add_argument('-logger', '--log-file', default='', help='log file name')
    parser.add_argument('--local-human', default=True, type='bool', help='log file name')   
    parser.add_argument('--display-examples', default=False, type='bool', help='')
    opt = parser.parse_args()
    
    # Set logging
    if opt['log_file'] is not '':
        logger = logging.getLogger('Evaluation: Seq2seq')
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        if 'log_file' in opt:
            logfile = logging.FileHandler(opt['log_file'], 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)
        logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))
        
    # Possibly build a dictionary (not all models do this).
    
    #assert opt['dict_file'] is None, '[ Put dict file ]'
            
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)
    
    run_eval(agent, opt, 'valid', write_log=True, logger=logger, generate=True, local_human=opt['local_human'])
    world.shutdown()
        
    
if __name__ == '__main__':
    main()





