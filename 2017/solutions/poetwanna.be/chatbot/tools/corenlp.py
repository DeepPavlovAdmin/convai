import requests
import urllib


class CoreNLPWrapper(object):
    """
    Provides basic interface to the StanfordCoreNLP server.
    For more advanced queries add new method or use `annotate` method directly.
    Refer to Stanford API: https://stanfordnlp.github.io/CoreNLP/annotators.html
    """

    def __init__(self, server_url):
        self.server_url = server_url

    def annotate(self, text, annotators, json_strict=False):
        """
        Parameter `annotators` may be series of multiple annotators
        separated with comas (e. g. `annotators="tokenize,ssplit,pos"`).
        """

        if isinstance(text, unicode):
            text = text.encode("utf8")

        r = requests.post(
            self.server_url,
            params={"annotators": annotators, "outputFormat": "json"},
            data=urllib.quote(text))
        return r.json(strict=json_strict)

    def tokenize(self, text):
        return self.annotate(text, annotators="tokenize")

    def sentence_split_and_tokenize(self, text):
        return self.annotate(text, annotators="ssplit")

    def pos_tag(self, text, parse=False):
        output = self.annotate(text, annotators="pos")
        if parse:
            output = [(token['originalText'], token['pos'])
                      for sentence in output["sentences"]
                      for token in sentence["tokens"]
                      ]
        return output

    def coref(self, text):
        return self.annotate(text, annotators="dcoref")

#    def clean_text(self, text):
#        text = text.replace('%', '')
#        text = text.replace('\x1b', '')
#        return text
