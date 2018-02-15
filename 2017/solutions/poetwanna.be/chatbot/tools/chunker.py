from __future__ import print_function

import config
from nltk import RegexpParser

'''
This simple noun phrase extractor comes from
https://stackoverflow.com/questions/38194579/extracting-noun-phrases-from-nltk-using-python
'''


def parse_tree(tags):
    grammar = r"""
         NBAR:
             {<NN.*|JJ.*|DT>*<NN.*>}
         NPIN:
             {<NBAR><IN><NBAR>}
             {<NBAR>}
         NPOS:
             {<NPIN><POS><NPIN>}
             {<NPIN>}
         NP:
             {<NPOS>}
     """
    chunker = RegexpParser(grammar)
    return chunker.parse(tags)


def get_noun_phrases4tree(tree):
    res = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        res.append(subtree.leaves())
    return res


def get_noun_phrases(tags, with_tags=False):
    if not tags or tags[0][1] == config.unknown_tag:
        return []
    tree = parse_tree(tags)
    phrases_tags = get_noun_phrases4tree(tree)
    if not with_tags:
        return [' '.join(zip(*l)[0]) for l in phrases_tags]
    return phrases_tags
