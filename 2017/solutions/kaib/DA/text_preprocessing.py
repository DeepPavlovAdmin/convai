import string
from data_helpers import clean_str
import json

import pdb
from nltk import word_tokenize 

# Read a file and split into lines
def readLines(filename):
    with open(filename) as data_file:
        read_dat = json.load(data_file, encoding = 'utf-8')['data']
        passages = []
        questions = []
        match = []
        article_idx = 0
        pssg_idx = 0
        
        for article in read_dat:
            # each paragraph is a context for the attached questions
            for paragraph in article['paragraphs']:
                passages.append(clean_str(paragraph['context']))
                    
                # each ouestion is an example
                for qa in paragraph['qas']:
                    questions.append(clean_str(qa['question']))
                    match.append([article_idx, pssg_idx])

                pssg_idx = pssg_idx + 1
            article_idx = article_idx + 1
    return passages, questions, match

 # Read a file and split into lines
def readLinesMS(filename):
    questions = []
    with open(filename) as data_file:
        read_dat = json.load(data_file, encoding = 'utf-8')['passage']
        print(read_dat[0])
    return questions


# Read a file and split into lines
def readLinesConvai(filename):
    with open(filename) as data_file:
        read_dat = json.load(data_file, encoding = 'utf-8')
        passages = []
        questions = []
        match = []
        article_idx = 0
        pssg_idx = 0

        for article in read_dat:
            # each paragraph is a context for the attached questions
            for userData in article['users']:
                if userData['userType'] == 'Human':
                     HumanUser = userData['id'] 
            context = (clean_str(article['context']))
            passages.append(context)
            for thread in article['thread']:
                if thread['userId'] == HumanUser:
                      questions.append(clean_str(thread['text']))
                      match.append([article_idx, pssg_idx])		

            pssg_idx = pssg_idx + 1
        article_idx = article_idx + 1
    return passages, questions, match


import re

def write_text(filename, dict):
    print(filename)
    writer = open(filename, 'w')
    for sent in dict:
        writer.write(sent + "\n")    
    writer.close()


def write_text_db(filename, dict):
    print(filename)
    writer = open(filename, 'w')
    for sent in dict:
        for sen in sent:
            writer.write(sen + "\t")
        writer.write("\n")
    writer.close()

def write_text_int(filename, dict):
    print(filename)
    writer = open(filename, 'w')
    for sent in dict:
        writer.write(','.join(map(str, sent)) + "\n")
    writer.close()


def write_text_noqmark(filename, dict):
    print(filename)
    writer = open(filename, 'w')
    for sent in dict:
        writer.write(re.sub(r"\\\?", "", sent).strip() + "\n")
    writer.close()

def squad_preprocessing():
    dataset = ['train', 'dev']
    for data in dataset:
        raw_file = './data/squad/' + data + '-v1.1.json'
        processed_q = './data/squad_' + data + '_q.txt'
        processed_p = './data/squad_' + data + '_p.txt'
        processed_no_qmark = './data/squad_' + data + '_no_qmark.txt'
        processed_idx = './data/squad_' + data + '_idx.json'
        processed_pq = './data/squad_' + data + '_pq.txt'

        querywithpassage = []
        passage_idx = 0
        
        passages, questions, match = readLines(raw_file)
        
        write_text(processed_p, passages)
        write_text(processed_q, questions)
        
        with open(processed_idx, 'w') as outfile:
            json.dump(match, outfile)
        for i, query in enumerate(questions):
            passage_idx = match[i][1]
            passage = passages[passage_idx]
            querywithpassage.append(query + '\t' + passage)

        write_text(processed_pq, querywithpassage)
    
def logs_preprocessing():
    dataset = ['cc', 'qa']
    for data in dataset:
        raw_data = '/data2/hwaranlee/convai/DA_cnn/data/log_' + data + '.txt'
        processed = '/data2/hwaranlee/convai/DA_cnn/data/log_processed_' + data + '.txt'
        processed_no_qmark = '/data2/hwaranlee/convai/DA_cnn/data/log_processed_' + data + '_no_qmark.txt'
        
        print(raw_data)
        questions = []
        f = open(raw_data, 'r')        
        lines = f.readlines()
        for line in lines:
            questions.append(clean_str(line))
        f.close()

        write_text(processed, questions)
        write_text_noqmark(processed_no_qmark, questions)        
        
def opensubs_preprocessing():
    dataset = ['dev', 'eval', 'train']
    for data in dataset:
        raw_data = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_' + data + '.txt'
        processed = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_processed_' + data + '.txt'
        processed_no_qmark = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_processed_' + data + '_no_qmark.txt'
        
        print(raw_data)
        questions = []
        f = open(raw_data, 'r')        
        lines = f.readlines()
        for line in lines:
            if line.find("U:") == 0:
                questions.append(clean_str(line[3:len(line)]))
        f.close()

        write_text(processed, questions)
        write_text_noqmark(processed_no_qmark, questions)
        print('# sentences : ' + str(len(questions)))

import random
        
def opensubs_prep_qdsep(): 
    # Q/D seperation
    # Both U/S are used
    dataset = ['train'] # [train', 'dev', 'eval']
    for data in dataset:
        raw_data = './data/opensubs_trial_data_' + data + '.txt'
        processed_p = './data/opensubs_trial_data_processed_' + data + '_q.txt' # question
        processed_d = './data/opensubs_trial_data_processed_' + data + '_d.txt' # declarative

        processed_a = './data/squad_' + data + '_p.txt'
        processed_ap = './data/squad_' + data + '_ap.txt'
        processed_aq = './data/squad_' + data + '_aq.txt'
        
        print(raw_data)
        questions = []
        declarative = []

        fp = open(processed_a, 'r')
        passline = fp.readlines()
        f = open(raw_data, 'r')        
        lines = f.readlines()
        for line in lines:
            if len(line) > 1 : # except "\n" 
                if line.find("?") == -1:
                    declarative.append(clean_str(line[3:len(line)]) + "\t" + clean_str(passline[random.randint(0, len(passline)-1)]))
                else:
                    questions.append(clean_str(line[3:len(line)])+ '\t' +  clean_str(passline[random.randint(0, len(passline)-1)]))
        fp.close()
        f.close()

        write_text(processed_p, questions)
        write_text(processed_d, declarative)

def msmarco_preprocessing():
    dataset = ['train', 'dev']
    for data in dataset:
        raw_file = './data/'+ data + '_v1.1.json' 
        processed = './data/log_processed_' + data + '.txt'
        processed_no_qmark = './data/log_processed_' + data + '_no_qmark.txt'

        questions = readLinesMS(raw_file)

        write_text(processed, questions)
        write_text_noqmark(processed_no_qmark, questions)
        print('# sentences : ' + str(len(questions)))


def convai_preprocessing():
    dataset = ['qa', 'cc']
    for data in dataset:
        raw_file = './data/convai_'+ data + '_passage.txt'
        processed = './data/convai_processed_' + data + '.txt'
        processed_no_qmark = './data/log_processed_' + data + '_no_qmark.txt'

        questions = []

        i = 0

        f = open(raw_file, 'r')
        lines = f.readlines()
        for line in lines:
            if (i%2==1 ) : # except "\n"
                questions.append(clean_str(lines[i-1]) +'\t' +  clean_str(line))
            i = i +1

        f.close()

        write_text(processed, questions)


def convai_total_preprocessing():
    dataset = ['train', 'test']
    for data in dataset:
        raw_file = './data/data_' + data + '_1501534800.json'
        processed_q = './data/data_processed_' + data + '_1501534800_q.txt'
        processed_p = './data/data_processed_' + data + '_1501534800_p.txt'
        processed_no_qmark = './data/data_processed_' + data + '_1501534800_no_qmark.txt'
        processed_idx = './data/data_processed_' + data + '_1501534800_idx.json'
        processed_pq = './data/data_processed_' + data + '_1501534800_pq.txt'

        querywithpassage = []
        passage_idx = 0

        passages, questions, match = readLinesConvai(raw_file)

        write_text(processed_p, passages)
        write_text(processed_q, questions)

        with open(processed_idx, 'w') as outfile:
            json.dump(match, outfile)
        for i, query in enumerate(questions):
            passage_idx = match[i][1]
            passage = passages[passage_idx]
            querywithpassage.append(query + '\t' + passage)

        write_text(processed_pq, querywithpassage)






if __name__ == '__main__':
    
    #squad_preprocessing()
    #logs_preprocessing()
    #opensubs_preprocessing()
    #opensubs_prep_qdsep()
    #msmarco_preprocessing()
    #convai_preprocessing()
    convai_total_preprocessing()










