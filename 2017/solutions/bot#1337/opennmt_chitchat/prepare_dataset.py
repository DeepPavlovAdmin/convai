import json
import re
from sys import argv, stderr
from nltk.tokenize import word_tokenize


def check_line(line):
    try:
        if line[0].upper() == line[1].upper():
            return False

        if line[-1].upper() == line[-2].upper():
            return False

        if len(line) > 7 and line[6].upper() == line[7].upper():
            return False

        if '[' in line or '(' in line:
            return False

        if '#' in line or ':' in line:
            return False


        if len(line.split(" ")) > 20:
            return False
    except IndexError:
        return False
    return True


def filter_line(line):
    line = re.sub(r"[-\"]", " ", line).replace('\r\n', "\n").replace("\n", " ").strip()
    line =  " ".join(word_tokenize(line.lower()))
    return line


if __name__ == '__main__':
    filename=argv[1]
    i = 0
    with open(filename, 'r') as f:
        prev_line = None
        current_line = None
        for line in f:
            line = line.strip()
            if check_line(line):
                prev_line = current_line
                current_line = filter_line(line)
                if prev_line:
                    i += 1
                    print("Processed: {}".format(i), file=stderr)
                    print("{}\t{}".format(prev_line, current_line))
                    prev_line = None
                    current_line = None
            else:
                prev_line = None
                current_line = None
