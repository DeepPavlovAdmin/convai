# Configuration file

class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

config_data = {
        'test_bot_token':'355748420:AAEpaGukZEeC1jFwvVU2TVf3d92fgq6VrKU',
        'bot_token':'5319E57A-F165-4BEC-94E6-413C38B4ACF9',
        'bot_endpoint':'https://ipavlov.mipt.ru/nipsrouter/',
        'bot_endpoint_backup':'https://koustuv.me/',
        # data endpoints
        # add your model endpoints here
        'data_base':'/root/convai/',
        'hred': {
            'twitter_model_prefix':'data/twitter_model/1489857182.98_TwitterModel',
            'reddit_model_prefix':'data/reddit_model/1485212785.88_RedditHRED',
            'twitter_dict_file':'data/twitter_model/Dataset.dict-5k.pkl',
            'reddit_dict_file':'data/reddit_model/Training.dict.pkl'
        },
        'de': {
            'reddit_model_prefix':'data/reddit-bpe5k_exp2/reddit_exp2',
            'reddit_data_file':'data/DE.dataset.pkl',
            'reddit_dict_file':'data/DE.dict.pkl',
            'convai-h2h_model_prefix': 'data/convai-h2h_exp1/convai-h2h_exp1',
            'convai-h2h_data_file':'data/round1_DE.dataset.pkl',
            'convai-h2h_dict_file':'data/round1_DE.dict.pkl'
        },
        'followup':{
            'model_prefix':'data/followup/',
            'dict_file':'data/followup/TrainingSmall.dict.pkl'
        },
        'candidate':{
            'dict_file':'data/candidate_dataset.txt'
        },
        'dumb':{
            'dict_file':'data/dumb_questions.json'
        },
        'topic':{
            'model_name':'/root/convai/data/yahoo_answers/fast.model.ep50.ng5.word2vec.bin',
            'dir_name':'data/yahoo_answers/',
            'top_k' : 2
        },
        "socket_port" : 8094,
        "ranker": {
            "model_short" : "/root/convai/ranker/models/short_term/0.641391/1510248853.21_Estimator_args.pkl",
            "model_long" : "/root/convai/ranker/models/long_term/1.4506/1510248853.21_short_term.0641391.151024885321_Estimator__args.pkl"
        },
        "stopwords" : ['all', 'whoever', 'go', 'whose',
            'to', 'help', 'helps', 'sorry', 'very', 'ha', 'haha',
            'hahahahahahaha', 'yourself', 'yourselves', '--', 'try', "I'll",
            'even', 'will', 'what', 'goes', 'new', 'never', 'here',
            "shouldn't", 'let', 'others', 'hers', "aren't", "I'd", "I'm",
            'ask', 'everybody', 'use', 'from', 'would', '&', 'destination',
            'next', 'few', 'themselves', 'today', 'more', 'knows', '&amp',
            'haha', 'excellent', 'glad', 'me', 'none', 'word', 'this', 'work',
            'can', 'learn', 'my', 'give', "didn't", 'hear', 'heard',
            'something', 'want', '.', ',', '?', '!', 'need', 'needs', 'how',
            'answer', 'instead', 'okay', 'may', 'man', 'a', 'so', 'pleasure',
            'talk', "that's", 'help', 'wut', 'over', 'still', 'its', 'perfect',
            'thank', 'fit', "he's", 'actually', 'better', 'ours', 'bye',
            'then', 'them', 'good', 'somebody', 'they', 'not', 'now', 'day',
            'several', 'name', 'always', 'did', 'someone', 'each', "isn't",
            'mean', 'everyone', 'hard', 'yeah', "we'd", 'our', 'out', "'",
            'since', "shouldn't", 'got', 'quite', 'put', 'could', 'keep',
            'isn', 'think', 'already', 'feel', '*', 'yourself', 'done', 'long',
            'another', "you're", '"', 'anyone', 'their', 'too', 'lot', 'that',
            'nobody', 'huh', 'herself', 'than', 'kind', 'future', 'were',
            'and', 'mind', 'mine', 'talking', 'have', 'need', 'seem', 'any',
            'answering', 'able', 'also', 'take', 'which', 'play', 'sure',
            'normal', 'who', 'most', 'plenty', 'nothing', 'why', "you'll",
            'show', 'fine', 'find', "hadn't", "don't", 'should', 'only',
            'going', 'hope', 'do', 'his', 'get', 'stop', 'him', 'bad', 'she',
            'where', 'theirs', 'see', 'are', 'worse', 'worst', 'best', 'lots',
            'wow', 'please', 'neither', 'nope', 'we', 'answers', 'news',
            'come', 'both', 'last', 'many', "can't", 'comment', 'tough',
            'seems', 'whatever', 'learning', 'forseeable', "it's", 'been',
            'whom', 'much', 'life', "what's", 'else', 'hmm', 'understand',
            'those', 'myself', 'look', 'these', 'hahaha', "wouldn't", 'is',
            'it', 'helped', 'itself', 'in', 'ready', 'if', 'perhaps', 'make',
            'same', 'get', 'gets', 'I', 'well', 'anybody', 'without', 'the',
            'yours', 'just', 'being', '-', 'thanks', 'questions', 'yep', 'yes',
            'hah', 'helping', 'had', 'has', 'gave', 'real', 'read', 'possible',
            'whichever', 'know', 'dare', 'like', 'tonight' 'night', 'whomever',
            'because', 'some', 'back', 'dear', 'curious', 'ourselves', 'for',
            'everything', 'does', 'either', 'be', 'by', 'on', 'about', 'ok',
            'anything', 'oh', 'of', 'or', 'seeing', 'own', 'into', 'down',
            'right', 'your', 'her', 'there', 'question', 'start', 'way', 'was',
            'himself', 'convai', 'ConvAI', 'but', 'hi', 'hear', 'ha', 'with',
            'dull', 'he', 'made', 'wish', 'up', 'us', 'am', 'an', 'as', 'at',
            'aw', 'home', 'happen', 'again', 'no', 'nah', 'when', 'other',
            'you', 'really', 'nice', 'alright', 'having', 'one'],
        "wh_words" : ['who', 'where', 'when', 'why', 'what', 'how',
            'whos', 'wheres', 'whens', 'whys', 'whats', 'hows']
}

def get_config():
    return dotdict(config_data)

