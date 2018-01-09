import argparse
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

def build_data(args):
    src = []
    tgt = []
    print('preprocessing data...')
    with open(args.datafile, 'r') as fin:
        current_src = []
        n_prev = args.n_prev_sent
        for line in tqdm(fin.readlines()):
            line = line.lstrip('-')
            line = line.strip()
            if len(current_src) < n_prev:
                current_src.append(' '.join(word_tokenize(line.lower())))
            else:
                src.append(' '.join(current_src))
                tgt.append(' '.join(word_tokenize(line.lower())))
                current_src = []
    
    src_train, src_test, tgt_train, tgt_test = train_test_split(src, tgt, test_size=0.2, random_state=42)

    print('saving data...')
    for filename, data in zip(['src.{}.train.txt', 'src.{}.test.txt', 'tgt.{}.train.txt', 'tgt.{}.test.txt' ], [src_train, src_test, tgt_train, tgt_test]):
        with open(filename.format(args.n_prev_sent), 'w', encoding='utf8') as fout:
            for line in data:
                fout.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datafile', type=str, default='open_subtitles2016.raw.en')
    parser.add_argument('--n_prev_sent', type=int, default=1, help='number of previous sentences')
    
    args = parser.parse_args()
    
    build_data(args)
