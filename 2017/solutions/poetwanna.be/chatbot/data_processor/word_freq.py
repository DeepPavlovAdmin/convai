from __future__ import print_function

import os
import sys
import re
import datetime
import subprocess
import json
import codecs

import nltk

from tools import progbar, tokenizer


def nltk_word_freq(infile, outfile, take_every=1, check_every=1000):
    """Count words using nltk sentence/word tokenizers"""

    print('Couting words in every %dth line...' % take_every)
    counter = nltk.probability.FreqDist()  # A better collections.Counter
    pbar = progbar.FileProgbar(infile, check_every=check_every)
    with codecs.open(infile, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            pbar.print_progress(i)
            if i % take_every != 0:
                continue
            # try:
            #     body = json.loads(line)['body']
            # except Exception, e:
            #     print(unicode(e))
            #     continue
            body = line
            for word in tokenizer.nltk_tokenize(body):
                counter[word] += 1
    print('\nSorting...')
    items = sorted(counter.items(), key=lambda (_, freq): -freq)
    print('Writing...')
    with codecs.open(outfile, 'w', 'utf-8') as f:
        for (word, freq) in items:
            f.write('%d %s\n' % (freq, word))
