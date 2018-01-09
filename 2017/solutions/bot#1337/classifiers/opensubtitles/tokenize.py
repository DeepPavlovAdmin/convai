import nltk
from nltk.tokenize import word_tokenize
from sys import argv


if __name__ == '__main__':
    filename=argv[1]

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            tokens = [t.lower() for t in word_tokenize(line)]
            normalized_line = " ".join(tokens)
            print(normalized_line)



