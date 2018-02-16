from QANet import QANet
from QANet import NAW_token as not_a_word_Str
from math import ceil
from random import choice
from answer_templates import answer_templates
from SQUADTalker import lower_if_needed


not_a_word_Char = 3


class AnswerBot:

    def __init__(self, model_file, glove_embs, glove_dict, negative, **kwargs):

        self.negative = negative
        self.model_file = model_file

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
                            train_unk=True,
                            negative=self.negative,
                            **kwargs)

        self.qa_net.load_params(self.model_file)

    def to_nums(self, ws):
        return [self.w_to_i.get(w, 0) for w in ws]

    def to_chars(self, w):
        return [1] + [self.c_to_i.get(c, 0) for c in w] + [2]

    def q_to_num(self, q):
        assert isinstance(q, list)
        return self.to_nums(q), map(self.to_chars, q)

    def prepare_question(self, q, x, q_num, q_char):
        assert isinstance(q, list)
        assert isinstance(x, list)

        x_num = self.to_nums(x)
        x_char = map(self.to_chars, x)

        def make_bin_feats(q, x):
            qset = set(q)
            return [w in qset for w in x]

        words = [[], q_num, x_num]
        chars = [q_char, x_char]
        bin_feats = make_bin_feats(q, x)

        if self.negative:
            words[2].append(self.not_a_word_Word)
            chars[1].append([1, not_a_word_Char, 2])
            bin_feats.append(False)
            x = x + [not_a_word_Str]

        return x, words, chars, bin_feats

    def get_answers(self, question, contexts, contexts_cased, beam=1):
        if not contexts:
            return []
        num_contexts = len(contexts)
        assert len(contexts_cased) == num_contexts

        q_words, q_chars = self.q_to_num(question)

        xs = []
        data = [[], [], []]

        for x in contexts:
            x, words, chars, bin_feats = self.prepare_question(
                question, x, q_words, q_chars)
            xs.append(x)
            data[0].append(words)
            data[1].append(chars)
            data[2].append(bin_feats)

        l, r, scr = self.qa_net._predict_spans(
            data, beam=beam, batch_size=num_contexts)

        answers = []
        all_contexts = u' '.join([u' '.join(c) if type(
            c) is list else c for c in contexts_cased])
        all_contexts_lower = lower_if_needed([[all_contexts]])[0][0]

        for i in range(num_contexts):
            answer = xs[i][l[i]:r[i]+1]
            # Try to retrieve the answer in original case
            answer_str = u' '.join(answer)
            pos = all_contexts_lower.find(answer_str)
            if pos != -1:
                answer = all_contexts[pos:pos + len(answer_str)].split(' ')
            answers.append((answer, scr[i]))

        return answers


class AnswerWrapper:

    def __init__(self):

        self.templates = answer_templates

    def wrap(self, answer, score):
        n = len(self.templates)
        k = int(n - ceil(score * n))
        cands = sum(self.templates[slice(max(0, k-1), min(10, k+2))], [])
        prefix, suffix = choice(cands)
        return prefix + answer + suffix
