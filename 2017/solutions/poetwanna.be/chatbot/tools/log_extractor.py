"""Extracts simple json logs"""
from __future__ import print_function
import codecs
import json
import os
import sys


if len(sys.argv) == 1:
    path = '/pio/lscratch/1/qa_nips_data/logs_dialogues/'
else:
    path = sys.argv[1]

# Typically it'll be config.cli_scoring_logs
logfiles = sorted(os.listdir(path))
print(logfiles)

for log in logfiles:
    print('-' * len(log))
    print(log)
    print('-' * len(log))
    with codecs.open(os.path.join(path, log), 'r') as f:
        for line in f:
            try:
                j = json.loads(line)
                if j.has_key('article'):
                    print(j['article']['text'].encode('utf-8'))
                else:
                    print('User:', j['dialogue_pair']['user'].encode('utf-8'))
                    print('Bot: ', j['dialogue_pair']['bot'].encode('utf-8'))
            except:
                print(j)
                raise
