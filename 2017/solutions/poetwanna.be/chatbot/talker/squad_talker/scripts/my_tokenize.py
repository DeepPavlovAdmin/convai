from nltk.tokenize import regexp_tokenize

'''
Split on whitespaces (excluding) and non-alphanumeric chars (including).
Two versions:
    - tokenize is faster
    - tokenize_with_ans_idx keeps track of answer index
'''


def tokenize_with_ans_idx(s, ans_idx=0):
    if type(s) is not unicode:
        s = s.decode('utf8')
    tokens = []
    buf = ''

    for i in range(len(s)):
        if ans_idx == 0:
            ans_start = len(tokens)
        ans_idx -= 1
        c = s[i]
        if c.isspace():
            if buf:
                tokens.append(buf)
            buf = ''
        elif not c.isalnum():
            if buf:
                tokens.append(buf)
            tokens.append(c)
            buf = ''
        else:
            buf += c
    if buf:
        tokens.append(buf)

    return tokens, ans_start


def tokenize(s):
    if type(s) is not unicode:
        s = s.decode('utf8')
    return regexp_tokenize(s, pattern='[^\W_]+|\S')
