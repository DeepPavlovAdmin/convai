import json
from sys import argv


if __name__ == '__main__':
    filename=argv[1]

    with open(filename) as f:
        dataset = json.load(f)

    for text in dataset["data"]:
        for par in text["paragraphs"]:
            for qa in par['qas']:
                print("__label__1 {}".format(qa['question']))
