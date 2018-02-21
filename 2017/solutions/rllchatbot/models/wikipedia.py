# Wikipedia Fact Extractor Model
# Given an entity, search wikipedia for that entity, and retrive the page
# From the retrieved page, extract a one line definition
# Use ElasticSearch API to query
# This script assumes ElasticSearch indices are downloaded and extracted in
# /var/lib/elasticsearch

import requests
import json
import nltk
import numpy as np


ELASTICSEARCH_PORT = 9200
ELASTICSEARCH_INDEX = 'enwiki'
BASE_LOC = 'http://localhost:{}/{}/page/'.format(ELASTICSEARCH_PORT,
        ELASTICSEARCH_INDEX)

def extract_article_sentence(query, num_sents=1):
    xq = {
            "query": {
                "match": {
                    "title": {
                        "query": query
                    }
                }
            }
        }
    res = requests.get(BASE_LOC + '_search', data=json.dumps(xq))
    res_data = res.json()
    page_hits = res_data['hits']['hits']
    max_score = res_data['hits']['max_score']
    # calculate the edit distance among the query and the results
    sims = [nltk.edit_distance(query, p['_source']['title']) for p in page_hits]
    top_ind = np.argmin(np.array(sims))
    # select the top page
    top_page = page_hits[top_ind]
    sents = nltk.sent_tokenize(top_page['_source']['opening_text'].encode('utf-8'))
    return ' '.join(sents[:num_sents])

if __name__=='__main__':
    print extract_article_sentence('McGill University')
