import os
import socket

from os.path import join, dirname, realpath


# Convai
convai_bot_id = ''  # TODO Set ID
# Testing:
# convai_server_url = 'https://ipavlov.mipt.ru/nipsrouter/'
# Main competition
# convai_sevrer_url = 'https://ipavlov.mipt.ru/nipsrouter/'
convai_server_url = os.environ.get('CONVAI_SERVER',
                                   'https://ipavlov.mipt.ru/nipsrouter/')
convai_bot_url = convai_server_url + convai_bot_id

convai_test_bot_url = 'https://ipavlov.mipt.ru/nipsrouter-alt/' + convai_bot_id

# Limits #dialogues per file for k-NN Talker
dialogue_limit_per_file = None

abs_path = lambda path: join(dirname(realpath(__file__)), path)
# Host-specific config

redis = os.environ.get('REDIS', 'redis')

glove_ver = '6B'  # do not change

if 'DATA_PATH' in os.environ:
    data_path = os.environ['DATA_PATH']
    swt_data_path = join(data_path, 'simple_wiki_data/')
    data_glove_path = data_path
    data_wiki_path = data_path
    data_mcr_path = data_path
    data_alice_path = data_path

wiki_paths = {
    'simple': {
        'index_dir': swt_data_path + '/simple',
        'positions_file': swt_data_path + '/simple/positions.txt',
        'tokens_file': None
    },
    'wiki4.5M': {
        'index_dir': data_wiki_path + '/wikipedia/wiki4.5M/',
        'positions_file': data_wiki_path + '/wikipedia/positions.txt',
        'tokens_file': data_wiki_path +
        '/wikipedia/en.wiki.tokenized.txt'
    },
}
default_wiki = 'wiki4.5M'

talker_weight = {
    'SBTTalker': 1.4,  # Same as SQUAD
    'SQUADTalker': 1.4,
    'SimpleWikiTalker': 1.25,
    'AbacusTalker': 1.0,  # harmless, only for math
    'CraftedKnnTalker': 0.8,  # NOT USED
    'MCRTalker': 0.9,  # upped from 0.8
    'BrainySmurfTalker': 0.3,  # same as trivia, they are followups
    'ChatterbotKnnTalker': 0.7,  # seems ok
    'TopicGuessTalker': 0.7,
    'TriviaTalker': 0.3,  # for a followupper this works fine
    'KnnTalker': 0.65,  # seems ok
    'AliceTalker': 0.5,
    'DbpediaTalker': 1.0,
    'GimmickTalker': 0.2,  # priority in certain scenarios
    'DefinitionsTalker': 1.0,  # priority for definitions, 1 is fine
    'ArticleAuxKnnTalker': 0.7,  # let's start with the same as chatterbot
}

preload_talkers = False or bool(os.environ.get('PRELOAD_TALKERS'))
celery_timeouts = False

# preload_talkers = True  # os.environ.get('PRELOAD_TALKERS')
# celery_timeouts = True

global_response_timeout = 60  # [s]
# [s] used for each talker routine and preprocessing
talker_respond_timeout = 5
# [s] used for each talker routine and preprocessing
talker_article_timeout = 15
wiki_timeout = 3  # [s]

# NOTE: It takes some 3,5GiB of RAM, will speed up caching k-nn vecs for data
word2vec_load_all_to_ram = False
word2vec_normalize = True
word2vec_floatx = 'float32'
word2vec_vecs_small = join(
    data_path, 'word2vec/word2vec_GoogleNews_200000.npy')
word2vec_txt_small = join(
    data_path, 'word2vec/word2vec_GoogleNews_200000.txt')
word2vec_vecs = join(data_path, 'word2vec/word2vec_GoogleNews_ALL.bin')
word2vec_txt = join(data_path, 'word2vec/word2vec_GoogleNews_ALL.txt')

# idf scaling options
swt_idf_scaling = True
squad_idf_scaling = True
dbpedia_idf_scaling = True
questionmark_bonus = 0.1

# enable / disable negative answers for squad model
squad_negative = False

# squad timeout in seconds
squad_timeout = 4.0

# MCR float type, don't use smaller types, it causes problems
mcr_floatx = 'float32'

# k-NN talkers
knn_idf = join(data_path, 'knn/idf.pkl')
knn_apply_idf = True
knn_method = 'ball_tree'
knn_floatx = 'float16'
assert knn_method in ['cdist', 'ball_tree']
knn_dialogues = {
    'chatterbot': join(data_path, 'knn/chatterbot/chatterbot-english.txt'),
    'crafted': join(data_path, 'knn/crafted/dialogue_pairs.txt'),
    'this_article': join(data_path,
                         'knn/this_article/this_article_with_news.txt'),
}
# NOTE If missing, those vecs will be computed and written to disk
knn_vecs = {
    'chatterbot': join(data_path, 'knn/chatterbot/chatterbot-english.txt.npy'),
    'crafted': join(data_path, 'knn/crafted/dialogue_vecs.npy'),
    'this_article': join(data_path,
                         'knn/this_article/this_article_with_news.txt.npy'),
}


dbpedia_resources = join(
    data_path, 'dbpedia/pagerank/top_100000_resources.txt')

hunspell_path = join(data_path, 'hunspell')
glove_path = join(data_glove_path, 'glove/')
glove_dict_path = glove_path + 'glove.' + glove_ver + '.wordlist.pkl'
glove_embs_path = glove_path + 'glove.' + glove_ver + '.300d.npy'

wiki_path = join(data_wiki_path, 'wikipedia/')
wiki_samples_articles = join(wiki_path, 'sample_articles.txt')
mcr_path = join(data_mcr_path, 'mcr_data/')
alice_path = join(data_alice_path, 'alice_data/')
stanford_path = join(data_path, 'stanford-postagger-2017-06-09/')
stanford_ner_path = join(data_path, 'stanford-ner-2017-06-09/')
squad_models_path = join(data_path, 'squad_models/')
thesaurus_path = join(data_path, 'dbpedia/thesaurus/')

trivia_path = join(data_path, 'trivia')
trivia_max_len = 160
trivia_cache_path = join(data_path, 'trivia.cache')

gimmick_langid_model = join(data_path, 'gimmick/langid_model.base64')
emoji_data = join(data_path, 'clean_one_char_emoji.txt')
polyglot_data_path = join(data_path, 'polyglot_data')


debug = (os.environ.get('DEBUG') or 'False').lower() in ['true', 't']
hacky_fast_start = os.environ.get('FAST_START')

cli_scoring_logs = join('/tmp', 'logs_dialogues')

unknown_tag = u'<UNK>'

telegram_bot = dict(
    TOKEN="XXX",
    CONTEXT_SIZE=3,
    REPLY_HIST_SIZE=20,
    LOGFILE="/tmp/log.txt"
)
