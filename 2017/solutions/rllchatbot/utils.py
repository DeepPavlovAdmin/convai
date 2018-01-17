import re
import random
import urlparse
import string


HTTP = re.compile('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')


def tokenize_utterance(utterance):
    """
    Process an utterance to be like in the training data to avoid out-of-vocab cases
    :param utterance: the text to process
    :type utterance: str
    :return: processed string
    """
    utterance = utterance.lower()
    res = HTTP.search(utterance)
    while res is not None:
        url_string = utterance[res.start():res.end()]
        url_tld = urlparse.urlparse(url_string).netloc
        url_tag = url_tld.replace('www.', '')
        for p in string.punctuation:
            url_tag = url_tag.replace(p, '_')
        url_tag = '<' + url_tag.strip() + '>'
        utterance = utterance.replace(url_string, url_tag)
        res = HTTP.search(utterance)

    utterance = utterance.replace(" (cont) ", "<cont>")
    utterance = utterance.replace('& lt', '<')
    utterance = utterance.replace('& gt', '>')
    utterance = utterance.replace('&lt;', '<')
    utterance = utterance.replace('&gt;', '>')
    utterance = utterance.replace('\'', ' \'')
    utterance = utterance.replace('"', ' " ')
    utterance = utterance.replace(";", " ")
    utterance = utterance.replace("`", " ")
    utterance = re.sub('\.+', '.', utterance)
    utterance = re.sub(',+', ',', utterance)
    utterance = utterance.replace('.', ' . ')
    utterance = utterance.replace('!', ' ! ')
    utterance = utterance.replace('?', ' ? ')
    utterance = utterance.replace(',', ' , ')
    utterance = utterance.replace('~', '')
    utterance = utterance.replace('-', ' - ')
    utterance = utterance.replace('*', ' * ')
    utterance = utterance.replace('(', ' ')
    utterance = utterance.replace(')', ' ')
    utterance = utterance.replace('[', ' ')
    utterance = utterance.replace(']', ' ')
    utterance = utterance.replace('>', '> ')
    utterance = utterance.replace('/', ' ')
    utterance = re.sub('\s+', ' ', utterance)
    utterance = utterance.strip()

    #utterance = utterance.replace('\xe2', ' <heart> ')

    # Convert @username to AT_USER
    utterance = re.sub('@[^\s]+', '<at>', utterance)

    # Remove hashtag sign from hashtags
    utterance = re.sub(r'#([^\s]+)', r'\1', utterance)

    # Replace numbers with <number> token
    utterance = re.sub(" \d+", " <number> ", utterance)

    utterance = utterance.replace('@', '')

    return unicode(utterance)


def detokenize_utterance(utterance, spacy_article=None):
    """
    Process a generated response to look more 'human'.
    :param utterance: text to process
    :type utterance: str
    :param spacy_article: list of words from the article with attribute `ent_type_`
    :return: processed text
    """
    if spacy_article:
        for tag in re.findall('<[^>]+>', utterance):
            if tag == '<number>':
                # select a number from the article
                instances = [w for w in spacy_article if w.ent_type_ == 'CARDINAL']
                if len(instances) > 0:
                    utterance.replace('<number>', random.choice(instances), 1)
            elif tag == '<at>':
                # select a number from the article
                instances = [w for w in spacy_article if w.ent_type_ == 'PERSON']
                if len(instances) > 0:
                    utterance.replace('<at>', random.choice(instances), 1)
            # TODO: add others...

    utterance = re.sub('<[^>]+>', '', utterance)  # remove all remaining <tags>

    utterance = utterance.replace(" '", "'")
    utterance = utterance.replace(" ,", ",")
    utterance = utterance.replace(" ?", "?")
    utterance = utterance.replace(" !", "!")
    utterance = utterance.replace(" -", "-")
    utterance = utterance.replace(" *", "*")

    # cap the first char
    utterance = utterance[0].capitalize() + utterance[1:]

    return utterance
