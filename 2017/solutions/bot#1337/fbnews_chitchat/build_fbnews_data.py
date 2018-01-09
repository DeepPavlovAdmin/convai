import os
import argparse
from tqdm import tqdm

from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split

def build_data(args):
    posts = pd.read_csv(args.posts_csv)
    comments = pd.read_csv(args.comments_csv)
    comments.dropna(axis=0, how='any', subset=['message', 'post_id'], inplace=True)
    posts.dropna(axis=0, how='any', subset=['message', 'post_id'], inplace=True)
    merged = pd.merge(posts, comments, on='post_id')
    merged = merged[['post_id', 'message_x', 'message_y']]
    posts = merged['message_x'].tolist()
    comments = merged['message_y'].tolist()
    merged = None

    print('tokenizing posts...')
    posts = list(map(lambda x: ' '.join(word_tokenize(x.lower())), tqdm(posts)))
    print('tokenizing comments...')
    comments = list(map(lambda x: ' '.join(word_tokenize(x.lower())), tqdm(comments)))

    posts_train, posts_test, comments_train, comments_test = train_test_split(posts, comments, test_size=0.2, random_state=42)
    print('saving data...')
    for filename, data in zip(['posts.train.txt', 'posts.test.txt', 'comments.train.txt', 'comments.test.txt'], [posts_train, posts_test, comments_train, comments_test]):
        with open(filename, 'w', encoding='utf8') as fout:
            for line in data:
                fout.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--posts_csv', type=str, default='fb_news_posts_20K.csv')
    parser.add_argument('--comments_csv', type=str, default='fb_news_comments_1000K.csv')
	
    args = parser.parse_args()
	
    build_data(args)
