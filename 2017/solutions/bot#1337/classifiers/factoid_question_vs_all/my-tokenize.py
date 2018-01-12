import nltk
from nltk.tokenize import word_tokenize
from sys import argv


if __name__ == '__main__':
    filename=argv[1]

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            label = line[:10]
            sent = line[11:]
            tokens = [t.lower() for t in word_tokenize(sent)]
            normalized_line = " ".join(tokens)
            print("{} {}".format(label, normalized_line))
