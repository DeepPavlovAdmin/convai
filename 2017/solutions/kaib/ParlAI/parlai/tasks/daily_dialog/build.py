# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.core.build_data as build_data
import os, sys, gzip, json, torch

import pdb


def parse_data(in_dir, out_dir, dataset='train'):

    # Read and Open files
    dial_dir = os.path.join(in_dir, 'dialogues_{}.txt'.format(dataset))
    emo_dir = os.path.join(in_dir, 'dialogues_emotion_{}.txt'.format(dataset))
    act_dir = os.path.join(in_dir, 'dialogues_act_{}.txt'.format(dataset))    
    # Open files
    in_dial = open(dial_dir, 'r')
    in_emo = open(emo_dir, 'r')
    in_act = open(act_dir, 'r')
    
    # Out json file    
    out_dir = os.path.join(out_dir, '{}.json'.format(dataset))
    
    data=[]
    for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
        seqs = line_dial.split('__eou__')[:-1]
        emos = line_emo.split()
        acts = line_act.split()
        
        seq_count = 0
        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)

        if seq_len != emo_len or seq_len != act_len:
            print("Different turns btw dialogue & emotion & acttion! ", line_count+1, seq_len, emo_len, act_len)
            sys.exit()

        dialog={}
        thread=[]        
        for seq, emo, act in zip(seqs, emos, acts):

            # Get rid of the blanks at the start & end of each turns
            seq = seq.strip().lower()
            utt={}
            utt['text']=seq
            utt['emo']=int(emo) # range 0~6
            utt['act']=int(act)-1 # range 0~3
            thread.append(utt)
        
        dialog['thread']=thread
        #dialog['topic']=topic
        data.append(dialog)

    in_dial.close()
    in_emo.close()
    in_act.close()
    
    with open(out_dir, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    stat(data)    
    
def stat(data):
    stat = torch.zeros(7,4).long()
    max_len= 0
    
    for i in range(len(data)):
        for j in range(len(data[i]['thread'])):
            stat[data[i]['thread'][j]['emo'], data[i]['thread'][j]['act']] += 1
            max_len = max(max_len, len(data[i]['thread'][j]['text'].split()))
    print(stat)
    print('max length = {}'.format(max_len))
    
    
def build(opt):

    data_path = os.path.join(opt['datapath'], 'DailyDialog')
    version = None
    
    if not build_data.built(data_path, version_string=version):
        print('[building data: ' + data_path + ']')

        if build_data.built(data_path):
            build_data.remove_dir(data_path)
        build_data.make_dir(data_path)

        fname = 'ijcnlp_dailydialog.zip'
        url = 'http://yanran.li/files/'
        
        # Download the data.
        # wget http://yanran.li/files/ijcnlp_dailydialog.zip
        # unzip ijcnlp_dailydialog.zip
        # unzip ijcnlp_dailydialog/*.zip
        
        
        #build_data.download(url, data_path, fname)
        #build_data.untar(data_path, fname)

        parse_data(os.path.join(data_path,'ijcnlp_dailydialog/train'), data_path, dataset='train')
        parse_data(os.path.join(data_path,'ijcnlp_dailydialog/validation'), data_path, dataset='validation')
        parse_data(os.path.join(data_path,'ijcnlp_dailydialog/test'), data_path, dataset='test')

        # Mark the data as built.
        build_data.mark_done(data_path, version_string=version)

