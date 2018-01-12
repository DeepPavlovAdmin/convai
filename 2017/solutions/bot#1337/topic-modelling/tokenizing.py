import nltk
import sys

from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from signal import signal, SIGPIPE, SIG_DFL

def convert_to_vw(text):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    lmtzr = WordNetLemmatizer()
    tokens = [t.lower() for t in tokenizer.tokenize(text)]
    id_ = 13371337
    processed = []
    for t in tokens:
        l = lmtzr.lemmatize(t)
        processed.append(l)
    counted = Counter(processed)
    res_str = str(id_)
    for k, v in counted.items():
        if v != 1:
            res_str = res_str + " {}:{}".format(k, v)
        else:
            res_str = res_str + " {}".format(k)
    return res_str


if __name__ == '__main__':
    fin = sys.stdin
    data = fin.read()
    print(convert_to_vw(data))



