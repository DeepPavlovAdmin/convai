# -*- coding: utf-8 -*-
import zmq, sys, json
from signal import signal, SIGPIPE, SIG_DFL


class ConnectionHandler:
    def __init__(self):
        signal(SIGPIPE, SIG_DFL)
        self.sock = zmq.Context().socket(zmq.REQ)
        self.sock.connect("tcp://127.0.0.1:5556")

    def __call__(self, data):
        self.sock.send_string(json.dumps(data))
        recieved = json.loads(str(self.sock.recv(), "utf-8"), encoding='utf-8', strict=False)
        recieved = [(row[0]['tgt'], row[0]['pred_score'], row[0]['src']) for row in recieved]
        return get_with_answers(recieved)


def get_with_answers(recieved):
    answers = []
    for _, _, src in recieved:
        tokens = src.split(' ')
        answer = []
        for token in tokens:
            features = token.split('￨')
            word = features[0]
            ans_tag = features[1]
            if ans_tag == 'B' or ans_tag == 'I':
                answer.append(word)
            elif answer:
                break
        answers.append(' '.join(answer))
    return [(recieved[i][0], answers[i], recieved[i][1]) for i in range(len(recieved))]

if __name__ == '__main__':
    fin = sys.stdin
    data = [{"src": line} for line in fin]

    connect = ConnectionHandler()
    received = connect(data)

    for target, answer, score in sorted(received, key=lambda x: x[2], reverse=True):
        print("{}\t{}\t{}".format(target, answer, score))
