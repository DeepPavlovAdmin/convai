# -*- coding: utf-8 -*-
import zmq, sys, json
import nltk
import nltk.tokenize
import re
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


def normalize(line):
    line = line.strip()
    tokens = [t.lower() for t in word_tokenize(line)]
    normalized_line = " ".join(tokens)
    return normalized_line


def detokenize(line):
    tokens = line.replace(" n't", "n't").split(' ')
    detokenizer = MosesDetokenizer()
    res = detokenizer.detokenize(tokens, return_str=True)
    res = res[0].upper() + res[1:]
    return res


if __name__ == '__main__':
    fin = sys.stdin
    data = [{"src": normalize(line)} for line in fin]

    url = argv[1]

    connect = ConnectionHandler(url)
    received = connect(data)

    for dst, score, src in sorted(received, key=lambda x: x[2], reverse=True):
        dst = detokenize(dst)
        print("{}\t{}\t{}".format(src, dst, score))
