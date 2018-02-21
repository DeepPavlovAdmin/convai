from __future__ import print_function


class PostingList:

    def __init__(self, content=None):
        self.last = 0
        self.b = []

        if content:
            for h in content.split():
                n = int(h, 16)
                self.b.append(n)
                self.last += n

    def append(self, n):
        d = n - self.last
        self.b.append(d)
        self.last = n

    def hex(self):
        return ' '.join([hex(x)[2:] for x in self.b])

    def to_list(self):
        res = []
        p = 0
        for n in self.b:
            p += n
            res.append(p)
        return res

    def __len__(self):
        return len(self.b)

if __name__ == "__main__":
    P = PostingList()
    L = [34, 55, 66, 776, 13334]

    for n in L:
        P.append(n)

    print(L, P.to_list())
    h = P.hex()
    print(h, PostingList(h).to_list())
