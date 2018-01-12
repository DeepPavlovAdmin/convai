import json
import re
from sys import argv


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
    return re.sub(r"[-\"]", " ", line).strip()


if __name__ == '__main__':
    filename=argv[1]

    with open(filename, 'r') as f:
        prev_prev_line = None
        prev_line = None
        current_line = None
        for line in f:
            line = line.strip()
            if check_line(line):
                prev_prev_line = prev_line
                prev_line = current_line
                current_line = filter_line(line)
                if prev_line:
                    print("{}\t{}".format(prev_line, current_line))
                if prev_prev_line:
                    print("{} {}\t{}".format(prev_prev_line, prev_line, current_line))
            else:
                prev_prev_line = None
                prev_line = None
                current_line = None
