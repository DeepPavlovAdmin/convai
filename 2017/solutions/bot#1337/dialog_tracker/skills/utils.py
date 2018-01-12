import itertools
import random
# TODO: Remove dependencies on from_* folders;
from from_opennmt_chitchat.get_reply import detokenize
from nltk import word_tokenize
from nltk.corpus import stopwords


def combinate_and_return_answer(arr):
    messages_product = list(itertools.product(*arr))
    msg_arr = random.sample(messages_product, 1)[0]
    msg = detokenize(" ".join(msg_arr))
    return msg


def get_stopwords_count(resp):
    return len(list(filter(lambda x: x.lower() in stopwords.words('english'), word_tokenize(resp))))
