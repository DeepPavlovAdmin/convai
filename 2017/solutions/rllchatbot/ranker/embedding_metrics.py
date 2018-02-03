from gensim.models import KeyedVectors
import numpy as np


print "loading word2vec embeddings..."
w2v = KeyedVectors.load_word2vec_format("/root/convai/data/GoogleNews-vectors-negative300.bin", binary=True)
# w2v = KeyedVectors.load_word2vec_format("/home/ml/nangel3/research/data/embeddings/GoogleNews-vectors-negative300.bin", binary=True)


def greedy_score(one, two):
    """Greedy matching between two texts"""
    dim = w2v.vector_size  # dimension of embeddings

    tokens1 = one.strip().split(" ")
    tokens2 = two.strip().split(" ")
    # X = np.zeros((dim,))  # array([ 0.,  0.,  0.,  0.])
    y_count = 0
    x_count = 0
    score = 0.0
    Y = np.zeros((dim,1))  # array([ [0.],  [0.],  [0.],  [0.] ])
    for tok in tokens2:    # for each token in the second text, add its column to Y
        if tok in w2v:
            y = np.array(w2v[tok])
            y /= np.linalg.norm(y)
            Y = np.hstack((Y, y.reshape((dim,1)) ))
            y_count += 1
    # Y ~ (dim, #of tokens in second text)
    # Y /= np.linalg.norm(Y)

    for tok in tokens1:  # for each token in the first text,
        if tok in w2v:
            x = np.array(w2v[tok])
            x /= np.linalg.norm(x)
            tmp = x.reshape((1,dim)).dot(Y)  # dot product with all other tokens from Y
            score += np.max(tmp)  # add the max value between this token and any token in Y
            x_count += 1

    # if none of the words in response or ground truth have embeddings, return zero
    if x_count < 1 or y_count < 1:
        return 0.0

    score /= float(x_count)
    return score


def extrema_score(one, two):
    """Extrema embedding score between two texts"""
    tokens1 = one.strip().split(" ")
    tokens2 = two.strip().split(" ")
    X = []
    for tok in tokens1:
        if tok in w2v:
            X.append(w2v[tok])
    Y = []
    for tok in tokens2:
        if tok in w2v:
            Y.append(w2v[tok])

    # if none of the words in text1 have embeddings, return 0
    if np.linalg.norm(X) < 0.00000000001:
        return 0.0

    # if none of the words in text2 have embeddings, return 0
    if np.linalg.norm(Y) < 0.00000000001:
        return 0.0

    xmax = np.max(X, 0)  # get positive max
    xmin = np.min(X, 0)  # get abs of min
    xtrema = []
    for i in range(len(xmax)):
        if np.abs(xmin[i]) > xmax[i]:
            xtrema.append(xmin[i])
        else:
            xtrema.append(xmax[i])
    X = np.array(xtrema)  # get extrema

    ymax = np.max(Y, 0)
    ymin = np.min(Y, 0)
    ytrema = []
    for i in range(len(ymax)):
        if np.abs(ymin[i]) > ymax[i]:
            ytrema.append(ymin[i])
        else:
            ytrema.append(ymax[i])
    Y = np.array(ytrema)  # get extrema

    score = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)
    return score


def average_score(one, two):
    """Average embedding score between two texts"""
    dim = w2v.vector_size # dimension of embeddings
    tokens1 = one.strip().split(" ")
    tokens2 = two.strip().split(" ")
    X = np.zeros((dim,))
    for tok in tokens1:
        if tok in w2v:
            X += w2v[tok]
    Y = np.zeros((dim,))
    for tok in tokens2:
        if tok in w2v:
            Y += w2v[tok]

    # if none of the words in text1 have embeddings, return 0
    if np.linalg.norm(X) < 0.00000000001:
        return 0.0

    # if none of the words in text2 have embeddings, return 0
    if np.linalg.norm(Y) < 0.00000000001:
        return 0.0

    X = np.array(X)/np.linalg.norm(X)
    Y = np.array(Y)/np.linalg.norm(Y)
    score = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)
    return score

