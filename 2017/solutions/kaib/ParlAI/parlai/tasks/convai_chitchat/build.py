# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.core.build_data as build_data
import os
import json
import random
import pdb

def build(opt):
    data_path = os.path.join(opt['datapath'], 'ConvAIChitChat')
    version = '1501534800'
    
    pdb.set_trace()
    
    if not build_data.built(data_path, version_string=version):
        print('[building data: ' + data_path + ']')

        if build_data.built(data_path):
            build_data.remove_dir(data_path)
        build_data.make_dir(data_path)

        fname = 'data_' + version + '.tar.gz'
        url = 'https://raw.githubusercontent.com/deepmipt/turing-data/master/' + fname
        build_data.download(url, data_path, fname)
        build_data.untar(data_path, fname)

        os.rename(os.path.join(data_path, 'data_train_' + version + '.json'), os.path.join(data_path, 'train.json'))
        os.rename(os.path.join(data_path, 'data_test_' + version + '.json'), os.path.join(data_path, 'test.json'))
        
        # Extract 10% of train.json into valid.json.
        with open(os.path.join(data_path, 'train.json')) as data_file:
            dialogs = json.load(data_file)
            random.seed(0)
            valid_dialogs_idxes = random.sample(range(len(dialogs)), round(len(dialogs) * 0.1))
            valid_dialogs = [dialogs[idx] for idx in valid_dialogs_idxes]
            train_dialogs_idxes = list(set(range(len(dialogs))) - set(valid_dialogs_idxes))
            train_dialogs = [dialogs[idx] for idx in train_dialogs_idxes]
            
        with open(os.path.join(data_path, 'valid.json'), 'w') as valid_data_file:
            json.dump(valid_dialogs, valid_data_file)
            
        with open(os.path.join(data_path, 'train.json'), 'w') as valid_data_file:
            json.dump(train_dialogs, valid_data_file)
            
        
        build_data.mark_done(data_path, version_string=version)
