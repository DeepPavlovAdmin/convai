# -*- coding: utf-8 -*-
import zmq, sys, json
from nltk import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from signal import signal, SIGPIPE, SIG_DFL
from sys import argv


class ConnectionHandler:
    def __init__(self, url):
        signal(SIGPIPE, SIG_DFL)
        self.sock = zmq.Context().socket(zmq.REQ)
        self.sock.connect(url)

    def __call__(self, data):
        self.sock.send_string(json.dumps(data))
        recieved = json.loads(str(self.sock.recv(), "utf-8"), encoding='utf-8', strict=False)
        recieved = [(row[0]['tgt'], row[0]['pred_score'], row[0]['src']) for row in recieved]
        return recieved


def map_brackets_fw(t):
    """Preprocessing for brackets"""
    if t == '(':
        return '-lrb-'
    if t == ')':
        return '-rrb-'
    return t

def map_brackets_bw(t):
    """Postprocessing for brackets"""
    if t == '-lrb-':
        return '('
    if t == '-rrb-':
        return ')'
    return t

def normalize(line):
    """Tokenize and lowercase"""
    line = line.strip()
    tokens = [t.lower() for t in word_tokenize(line)]
    tokens = map(lambda x: map_brackets_fw(x), tokens)
    normalized_line = ' '.join(tokens).replace(' . ', ' ')
    return normalized_line


def detokenize(line):
    tokens = line.replace(" n't", "n't").split(' ')
    tokens = list(map(lambda x: map_brackets_bw(x), tokens))
    detokenizer = MosesDetokenizer()
    res = detokenizer.detokenize(tokens, return_str=True)
    res = res[0].upper() + res[1:]
    return res

def split_text_on_chunks(text, max_len=75, overlap=15):
    """
    Args:
        text: input text
        max_len: max chunk length (in words)
        overlap: length of chunks overlapping

    Returns:
        list of chunks
    """
    chunks = []
    tokens = normalize(text).split(' ')
    for i in range(0, len(tokens), overlap):
        chunks.append(' '.join(tokens[i: i + max_len]))
    return chunks

if __name__ == '__main__':
    fin = sys.stdin
    # input source text
    # output - summaries for text chunks
    data = [{"src": chunk} for line in fin for chunk in split_text_on_chunks(line)]

    url = argv[1]

    connect = ConnectionHandler(url)
    received = connect(data)

    for dst, score, src in sorted(received, key=lambda x: x[2], reverse=True):
        dst = detokenize(dst)
        print("{}\t{}\t{}".format(src, dst, score))
