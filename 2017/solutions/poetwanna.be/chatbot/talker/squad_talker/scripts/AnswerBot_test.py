import sys
sys.path.append('../')

from my_tokenize import tokenize
from QANet import QANet


not_a_word_Char = 3
not_a_word_Str = '<not_a_word>'


class AnswerBot:

    def __init__(self, model_file, glove_embs, glove_dict,
                 glove_ver, negative, **kwargs):

        self.negative = negative
        self.model_file = model_file
        self.glove_ver = glove_ver

        self.words = glove_dict
        self.w_to_i = {v: k for (k, v) in enumerate(self.words)}

        self.not_a_word_Word = self.w_to_i[not_a_word_Str]

        self.glove_embs = glove_embs
        self.voc_size = self.glove_embs.shape[0]

        self.chars = [unichr(i) for i in range(128)]
        self.c_to_i = {v: k for (k, v) in list(enumerate(self.chars))}

        self.qa_net = QANet(self.voc_size,
                            emb_init=self.glove_embs,
                            skip_train_fn=True,
                            negative=self.negative,
                            **kwargs)

        self.qa_net.load_params(self.model_file)

    def prepare_question(self, q, x):
        assert type(q) is type(x)
        assert type(q) in [str, unicode, list]

        def to_nums(ws):
            return [self.w_to_i.get(w, 0) for w in ws]

        def to_chars(w):
            return [1] + [self.c_to_i.get(c, 0) for c in w] + [2]

        def make_words(q, x):
            return [[], to_nums(q), to_nums(x)]

        def make_chars(q, x):
            return [map(to_chars, q), map(to_chars, x)]

        def make_bin_feats(q, x):
            qset = set(q)
            return [w in qset for w in x]

        def lower_if_needed(l):
            if self.glove_ver == '6B':
                return [w.lower() for w in l]
            return l

        if type(q) is not list:
            q = lower_if_needed(tokenize(q))
            x = lower_if_needed(tokenize(x))

        neg = self.negative
        if neg and x[-1] == not_a_word_Str:
            neg = False

        data = make_words(q, x), make_chars(q, x), make_bin_feats(q, x)

        if neg:
            data[0][2].append(self.not_a_word_Word)
            data[1][1].append([1, not_a_word_Char, 2])
            data[2].append(False)
            x.append(not_a_word_Str)

        return (q, x) + data

    def get_answers(self, questions, contexts, beam=1):
        num_contexts = len(contexts)
        assert len(questions) == num_contexts

        sample = []
        data = [[], [], []]

        for i in range(num_contexts):
            q = questions[i]
            x = contexts[i]
            q, x, words, chars, bin_feats = self.prepare_question(q, x)
            sample.append([q, x])
            data[0].append(words)
            data[1].append(chars)
            data[2].append(bin_feats)

        l, r, scr = self.qa_net._predict_spans(
            data, beam=beam, batch_size=num_contexts)

        answers = []
        for i in range(num_contexts):
            answer = sample[i][1][l[i]:r[i]+1]
            answers.append((answer, scr[i]))

        return answers
