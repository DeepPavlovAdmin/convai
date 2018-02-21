from __future__ import print_function

import os
import cdb
import config

from talker.simple_wiki_talker.swt_utils import lower_string, key_title


DATA_PATH = config.swt_data_path

definitions = {}
redirect = {}
wiktionary_definitions = {}


def load_data():
    global definitions
    global wiktionary_definitions
    global redirect

    for x in open(os.path.join(DATA_PATH, 'simple_wiki_fs.txt')):
        x = x.strip()
        a, b = x.split('\t')
        definitions[lower_string(a)] = b

    for x in open(os.path.join(DATA_PATH, 'simple_wiki_redirect.txt')):
        a, b = lower_string(x).split()
        redirect[key_title(a)] = key_title(b)

    wiktionary_definitions = cdb.init(
        os.path.join(DATA_PATH, 'wiktionary.cdb'))


def get_definition(word):
    word = lower_string(word)
    if word in redirect and redirect[word] in definitions:
        print("REDIRECTING", word, "-->", redirect[word])
        return definitions[redirect[word]]
    return definitions.get(word, '')


def get_wiktionary_definition(word, preferred_pos='noun', lang='English'):
    if type(word) == unicode:
        word = word.encode('utf8')

    definition_string = wiktionary_definitions.get(word)
    if definition_string is None:
        return ''

    definition = eval(definition_string)

    languages = set(lan for pos, lan in definition)
    if lang in languages:
        poss = set(pos for (pos, lan) in definition if lan == lang)
        if preferred_pos in poss:
            return definition[preferred_pos, lang]
        for p in poss:
            return definition[p, lang]

    lans = set()
    lan_d = []
    for pos, lan in definition:
        if not pos in lan_d:
            d = definition[pos, lan]
            if d.strip() and '#' not in d and len(d) != 1:
                lan_d.append((lan, d))
                lans.add(lan)
    if len(lan_d) > 1:
        meanings = '. '.join(' In ' + lan + ': ' + d.replace('#' + lang, ' ')
                             for (lan, d) in lan_d)
        return word + ' has different meanings.' + meanings
    elif len(lan_d) == 1:
        return lan_d[0][1]
    return ''
