import config
from os.path import join
from nltk import word_tokenize
from nltk.tag import StanfordNERTagger


print 'Loading Stanford NER Tagger data...'
classifier = join(config.stanford_ner_path,
                  'classifiers/english.all.3class.distsim.crf.ser.gz')
jar = join(config.stanford_ner_path, 'stanford-ner.jar')
snt = StanfordNERTagger(classifier, jar)
print 'Done'


def tag_sentence(sentence):
    return snt.tag(word_tokenize(sentence))
