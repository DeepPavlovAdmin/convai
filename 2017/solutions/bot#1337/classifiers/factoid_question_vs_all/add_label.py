from sys import argv


if __name__ == '__main__':
    filename = argv[1]
    label = argv[2]
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            print("{} {}".format(label, line))

