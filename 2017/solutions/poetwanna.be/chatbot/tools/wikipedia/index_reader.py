from posting_list_hex import PostingList


class IndexReader:

    def __init__(self, index_name, index_positions_name):
        self.index_file = open(index_name, 'r')
        self.positions = {}
        self.cache = {}

        for x in open(index_positions_name):
            term, pos = x.split()
            self.positions[term] = int(pos)

    def get(self, term):
        if term not in self.positions:
            return set()
        if term in self.cache:
            return self.cache[term]

        pos = self.positions[term]
        self.index_file.seek(pos)
        line = self.index_file.readline().split()
        line = ' '.join(line[1:])
        result = PostingList(line).to_list()
        self.cache[term] = result
        return result
