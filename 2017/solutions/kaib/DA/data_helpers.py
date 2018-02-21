import numpy as np
import re
import itertools
from collections import Counter

#from random import randrange
import random
import pdb

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_data_and_labels_qd(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = []
    for pos in positive_data_file:
        list_temp = list(open(pos, "r").readlines())
        positive_examples = positive_examples + list_temp
    
    negative_examples = []
    for neg in negative_data_file:
        #list_temp = list(open(neg, "r").readlines())
        list_temp = open(neg, "r").readlines()
        random.shuffle(list_temp)
        ndata = int(min(len(list_temp), 887560/2))
        negative_examples = negative_examples + list_temp[0:ndata]
    
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_qa(positive_data_file, negative_data_file, nutral_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = []
    for pos in positive_data_file:
        list_temp = list(open(pos, "r").readlines())
        positive_examples = positive_examples + list_temp

    negative_examples = []
    for neg in negative_data_file:
        #list_temp = list(open(neg, "r").readlines())
        list_temp = open(neg, "r").readlines()
        random.shuffle(list_temp)
        ndata = int(min(len(list_temp), 887560/2))
        negative_examples = negative_examples + list_temp[0:ndata]


    nutral_examples = []
    for nut in nutral_data_file:
        list_temp = list(open(pos, "r").readlines())
        nutral_examples = nutral_examples + list_temp


    # Split by words
    x_text = positive_examples + negative_examples +nutral_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1, 0] for _ in positive_examples]
    negative_labels = [[1, 0, 0] for _ in negative_examples]
    nutral_labels = [[0, 0, 1] for _ in nutral_examples]
    y = np.concatenate([positive_labels, negative_labels, nutral_labels], 0)
    return [x_text, y]

def checkMatch(passage, query):
    wordmatch = []
    wordquery = query.split(" ")
    for onewd in wordquery:
        if(passage.find(onewd) < 0):
             wordmatch.append(0)
        else:
             wordmatch.append(1)
    return wordmatch

def load_data_and_labels_ev(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    positive_examples = []
    match = []
    list_temp = list(open(positive_data_file, "r").readlines())
    for line in list_temp:
        word_temp = line.split("\t")
        match.append(checkMatch(word_temp[1], word_temp[0]))
        positive_examples.append(word_temp[0])


    negative_examples = []
    list_temp = open(negative_data_file, "r").readlines()
    for line in list_temp:
        word_temp = line.split("\t")
        match.append(checkMatch(word_temp[1], word_temp[0]))
        negative_examples.append(word_temp[0])

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y, match]


def load_data_and_labels_em(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = []
    match = []
    for pos in positive_data_file:
        list_temp = list(open(pos, "r").readlines())
        #positive_examples = positive_examples + list_temp
        for line in list_temp:
            word_temp = line.split("\t")
            match.append(checkMatch(word_temp[1], word_temp[0]))
            positive_examples.append(word_temp[0])


    negative_examples = []
    for neg in negative_data_file:
        list_temp = open(neg, "r").readlines()
        random.shuffle(list_temp)
        ndata = int(min(len(list_temp), 88025/2))
        for line in list_temp[0:ndata]:
            word_temp = line.split("\t")
            match.append(checkMatch(word_temp[1], word_temp[0]))
            negative_examples.append(word_temp[0])


        #ndata = int(min(len(list_temp), 887560/2))
        #negative_examples = negative_examples + list_temp[0:ndata]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    #for sent in xtmp_text:
    #    x_text.append(clean_str(sent[0]))
    #    match.append(sent[1].split(","))

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y, match]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def batch_iter2(data, data2, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data2 = np.array(data2)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch2 in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            shuffled_data2 = data2
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (shuffled_data[start_index:end_index], shuffled_data2[start_index:end_index])

