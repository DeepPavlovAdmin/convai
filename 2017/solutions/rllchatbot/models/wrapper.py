import zmq
import cPickle
import logging
import numpy as np
import re

import lasagne
import theano
from dual_encoder.model import Model as DE_Model
from hredqa.hred_pytorch import HRED_QA

import hred.search as search
import utils
from hred.dialog_encdec import DialogEncoderDecoder
from hred.state import prototype_state
from candidate import CandidateQuestions
from alicebot.nlg_alicebot import NLGAlice
import json
import random
import requests
import delegator
import codecs
from nltk import sent_tokenize
from gensim.models.keyedvectors import KeyedVectors
import config
import os
import cPickle as pkl
conf = config.get_config()


#logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)

NQG_ENDURL = 'http://localhost:8080'
DRQA_ENDURL = 'http://0.0.0.0:8888'
FASTTEXT_DIR = '/root/convai/models/fastText/'


class Model_Wrapper(object):
    """
    Super class for all Model wrappers
    """

    def __init__(self, model_prefix, name):
        """
        Default constructor
        :param model_prefix: path to the model files
        :param name: model name
        """
        self.model_prefix = model_prefix
        self.name = name
        self.speaker_token = ['<first_speaker>', '<second_speaker>']
        if self.name == 'hred-reddit':
            self.speaker_token = ['<speaker_1>', '<speaker_2']

    def _format_to_model(self, text, context_length):
        text = utils.tokenize_utterance(text)
        text = '%s %s </s>' % (
            self.speaker_token[context_length % 2], text.strip().lower())
        return text

    def _format_to_user(self, text):
        text = utils.detokenize_utterance(text)
        # strip, split, join to remove extra spaces
        return ' '.join(text.strip().split())

    def get_response(self, user_id='', text='', context=None, article='', **kwargs):
        """
        Generate a new response, and add it to the context
        :param article:
        :param user_id: id of the person we chat with
        :param text: the new utterance we just received
        :type text: str
        :param context: queue of conversation history: sliding window of most recent utterances
        :type context: array
        :type kwargs extra params
        :return: the generated response as well as the new context
        """
        pass  # TO BE IMPLEMENTED IN SUB-CLASSES

    def preprocess(self, user_id='', article=None, **kwargs):
        """
        Preprocess before model responses, needed for some models (NQG)
        """
        pass  # OPTIONAL, may or may not implement this

    def isMatch(self, text='', **kwargs):
        """
        For some models, it is better to check the intent using some pre-set
        regexes. Use this to do that.
        """
        pass  # OPTIONAL


class HRED_Wrapper(Model_Wrapper):
    """
    GENERIC GENERATIVE MODEL
    """

    def __init__(self, model_prefix, dict_file, name):
        # Load the HRED model.
        super(HRED_Wrapper, self).__init__(model_prefix, name)
        state_path = '%s_state.pkl' % model_prefix
        model_path = '%s_model.npz' % model_prefix

        state = prototype_state()
        with open(state_path, 'r') as handle:
            state.update(cPickle.load(handle))
        state['dictionary'] = dict_file
        logging.info('Building %s model...' % name)
        self.model = DialogEncoderDecoder(state)
        logging.info('Building sampler...')
        self.sampler = search.BeamSampler(self.model)
        logging.info('Loading model...')
        self.model.load(model_path)
        logging.info('Model built (%s).' % name)

    # must contain this method for the bot
    def get_response(self, user_id='', text='', context=None, article='', **kwargs):
        logging.info('--------------------------------')
        logging.info('Generating HRED response for user %s.' % user_id)
        text = self._format_to_model(text, len(context))
        context.append(text)
        logging.info('Using context: %s' % ' , '.join(list(context)))

        samples, costs = self.sampler.sample(
            [' '.join(list(context))],
            ignore_unk=True,
            verbose=False,
            return_words=True
        )
        response = samples[0][0].replace('@@ ', '').replace('@@', '')
        # remove all tags to avoid having <unk>
        response = self._format_to_user(response)
        # add appropriate tags to the response in the context
        context.append(self._format_to_model(response, len(context)))
        logging.info('Response: %s' % response)
        return response, context


class Dual_Encoder_Wrapper(Model_Wrapper):
    """
    GENERIC RETRIEVER MODEL
    """

    def __init__(self, model_prefix, data_fname, dict_fname, name, n_resp=10000):
        super(Dual_Encoder_Wrapper, self).__init__(model_prefix, name)

        try:
            with open('%s_model.pkl' % model_prefix, 'rb') as handle:
                self.model = cPickle.load(handle)
        except Exception as e:
            logging.error("%s\n ERROR: couldn't load the model" % e)
            logging.info("Will create a new one with pretrained parameters")
            # Loading old arguments
            with open('%s_args.pkl' % model_prefix, 'rb') as handle:
                old_args = cPickle.load(handle)

            logging.info("Loading data...")
            with open('%s' % data_fname, 'rb') as handle:
                train_data, val_data, test_data = cPickle.load(handle)
            data = {'train': train_data, 'val': val_data, 'test': test_data}
            # W is the word embedding matrix and word2idx, idx2word are dictionaries
            with open('%s' % dict_fname, 'rb') as handle:
                word2idx, idx2word = cPickle.load(handle)
            W = np.zeros(shape=(len(word2idx), old_args.emb_size))
            for idx in idx2word:
                W[idx] = np.random.uniform(-0.25, 0.25, old_args.emb_size)
            logging.info("W.shape: %s" % (W.shape,))

            logging.info("Creating model...")
            self.model = self._create_model(
                data, W, word2idx, idx2word, old_args)

            logging.info("Set the learned weights...")
            with open('%s_best_weights.pkl' % model_prefix, 'rb') as handle:
                params = cPickle.load(handle)
                lasagne.layers.set_all_param_values(self.model.l_out, params)
            with open('%s_best_M.pkl' % model_prefix, 'rb') as handle:
                M = cPickle.load(handle)
                self.model.M.set_value(M)
            with open('%s_best_embed.pkl' % model_prefix, 'rb') as handle:
                em = cPickle.load(handle)
                self.model.embeddings.set_value(em)

        with open('%s_timings.pkl' % model_prefix, 'rb') as handle:
            timings = cPickle.load(handle)
            # load last timings (when no improvement was done)
            self.model.timings = timings
        logging.info("Model loaded.")

        with open("%s_r-encs.pkl" % model_prefix, 'rb') as handle:
            self.cached_retrieved_data = cPickle.load(handle)
        self.n_resp = n_resp

    def _create_model(self, data, w, word2idx, idx2word, args):
        return DE_Model(
            data=data,
            W=w.astype(theano.config.floatX),
            word2idx=word2idx,
            idx2word=idx2word,
            save_path=args.save_path,
            save_prefix=args.save_prefix,
            max_seqlen=args.max_seqlen,  # default 160
            batch_size=args.batch_size,  # default 256
            # Network architecture:
            encoder=args.encoder,  # default RNN
            hidden_size=args.hidden_size,  # default 200
            n_recurrent_layers=args.n_recurrent_layers,  # default 1
            is_bidirectional=args.is_bidirectional,  # default False
            dropout_out=args.dropout_out,  # default 0.
            dropout_in=args.dropout_in,  # default 0.
            # Learning parameters:
            patience=args.patience,  # default 10
            optimizer=args.optimizer,  # default ADAM
            lr=args.lr,  # default 0.001
            lr_decay=args.lr_decay,  # default 0.95
            fine_tune_W=args.fine_tune_W,  # default False
            fine_tune_M=args.fine_tune_M,  # default False
            # NTN parameters:
            use_ntn=args.use_ntn,  # default False
            k=args.k,  # default 4
            # Regularization parameters:
            penalize_emb_norm=args.penalize_emb_norm,  # default False
            penalize_emb_drift=args.penalize_emb_drift,  # default False
            emb_penalty=args.emb_penalty,  # default 0.001
            penalize_activations=args.penalize_activations,  # default False
            act_penalty=args.act_penalty  # default 500
        )

    def get_response(self, user_id='', text='', context=None, article='', **kwargs):
        logging.info('--------------------------------')
        logging.info('Generating DE response for user %s.' % user_id)
        text = self._format_to_model(text, len(context))
        context.append(text)
        logging.info('Using context: %s' % ' '.join(list(context)))

        # TODO: use tf-idf as a pre-filtering step to only retrive from `self.n_resp`
        # for now, sample `self.n_resp` randomly without replacement
        response_set_idx = range(len(self.cached_retrieved_data['r']))
        np.random.shuffle(response_set_idx)
        response_set_idx = response_set_idx[:self.n_resp]
        response_set_str = [self.cached_retrieved_data['r'][i]
                            for i in response_set_idx]
        response_set_embs = [self.cached_retrieved_data['r_embs'][i]
                             for i in response_set_idx]

        cached_retrieved_data = self.model.retrieve(
            context_set=[' '.join(list(context))],
            response_set=response_set_str,
            response_embs=response_set_embs,
            k=1, batch_size=1, verbose=False
        )
        response = cached_retrieved_data['r_retrieved'][0][0].replace(
            '@@ ', '').replace('@@', '')

        # remove all tags to avoid having <unk>
        response = self._format_to_user(response)
        # add appropriate tags to the response in the context
        context.append(self._format_to_model(response, len(context)))
        logging.info('Response: %s' % response)
        return response, context


class Human_Imitator_Wrapper(Dual_Encoder_Wrapper):
    """
    RETRIEVE THE MOST LIKELY HUMAN RESPONSE FROM CONVAI ROUND1 CHATS
    NOTE: SUBCLASS OF DUAL_ENCODER_WRAPPER
    """

    def __init__(self, model_prefix, data_fname, dict_fname, name, n_resp=10000):
        super(Human_Imitator_Wrapper, self).__init__(
            model_prefix, data_fname, dict_fname, name, n_resp)

    def get_response(self, user_id='', text='', context=None, article='', **kwargs):
        logging.info('--------------------------------')
        logging.info('Generating DE (human) response for user %s.' % user_id)
        text = utils.tokenize_utterance(text.strip().lower())
        context.append(text)
        logging.info('Using context: %s' % ' '.join(list(context)))

        response_set_str = self.cached_retrieved_data['r']
        response_set_embs = self.cached_retrieved_data['r_embs']

        cached_retrieved_data = self.model.retrieve(
            context_set=[' </s> '.join(list(context))],
            response_set=response_set_str,
            response_embs=response_set_embs,
            k=1, batch_size=1, verbose=False
        )
        response = cached_retrieved_data['r_retrieved'][0][0]

        # remove all tags to avoid having <unk>
        response = self._format_to_user(response)
        # add appropriate tags to the response in the context
        context.append(response)
        logging.info('Response: %s' % response)
        return response, context


class HREDQA_Wrapper(Model_Wrapper):
    """
    GENERATE A FOLLOWUP QUESTION. ie: WHAT? WHY? YOU?
    BAD MODEL IN GENERAL...
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(HREDQA_Wrapper, self).__init__(model_prefix, name)

        self.model = HRED_QA(
            dictionary=dict_fname,
            encoder_file='{}encoder_5.model'.format(model_prefix),
            decoder_file='{}decoder_5.model'.format(model_prefix),
            context_file='{}context_5.model'.format(model_prefix)
        )

    def _get_sentences(self, context):
        sents = [re.sub('<[^>]+>', '', p) for p in context]
        return sents

    def _format_to_user(self, text):
        text = super(HREDQA_Wrapper, self)._format_to_user(text)
        if not text.endswith('?'):
            text = text + ' ?'
        # strip, split, join to remove extra spaces
        return ' '.join(text.strip().split())

    def get_response(self, user_id='', text='', context=None, article='', **kwargs):
        logging.info('------------------------------------')
        logging.info('Generating Followup question for user %s.' % user_id)
        text = self._format_to_model(text, len(context))
        context.append(text)
        logging.info('Using context: %s' % ' '.join(list(context)))

        response = self.model.evaluate(
            self.model.encoder_model,
            self.model.decoder_model,
            self.model.context_model,
            self._get_sentences(context)
        )
        response = ' '.join(response)
        response = self._format_to_user(response)
        context.append(self._format_to_model(response, len(context)))
        return response, context


class CandidateQuestions_Wrapper(Model_Wrapper):
    """
    ASK A PREDEFINED QUESTION ABOUT AN ENTITY IN THE ARTICLE
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(CandidateQuestions_Wrapper, self).__init__(model_prefix, name)
        # Use these questions if no suitable questions are found
        # TODO: do not hardcode these, use a dictionary
        self.dict_fname = dict_fname
        self.canned_questions = ["That's a short article, don't you think? Not sure what's it about.",
                                 "Apparently I am too dumb for this article. What's it about?"]
        self.models = {}
        self.canned_freq_user = {}   # Only allow one canned response per user

    def preprocess(self, chat_id='', article_text='', **kwargs):
        logging.info("Preprocessing CandidateQuestions")
        assert isinstance(article_text, basestring)
        self.models[chat_id] = CandidateQuestions(
            article_text, self.dict_fname)
        self.canned_freq_user[chat_id] = 0

    def _get_sentences(self, context):
        sents = [re.sub('<[^>]+>', '', p) for p in context]
        return sents

    def _format_to_user(self, text):
        text = super(HREDQA_Wrapper, self)._format_to_user(text)
        if not text.endswith('?'):
            text = text + ' ?'
        # strip, split, join to remove extra spaces
        return ' '.join(text.strip().split())

    def get_response(self, chat_id='', text='', context=None, article='', **kwargs):
        logging.info('------------------------------------')
        logging.info('Generating candidate question for chat %s.' % chat_id)
        text = self._format_to_model(text, len(context))
        logging.info(text)
        context.append(text)

        if chat_id in self.models:
            response = self.models[chat_id].get_response()
            if len(response) < 1 and self.canned_freq_user[chat_id] < 1:
                # select canned response
                response = random.choice(self.canned_questions)
                self.canned_freq_user[chat_id] += 1
        else:
            response = 'What is this article about?'  # default
        context.append(self._format_to_model(response, len(context)))
        return response, context


class DumbQuestions_Wrapper(Model_Wrapper):
    """
    IF USER ASKED A SIMPLE QUESTION RETURN A PREDEFINED ANSWER
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(DumbQuestions_Wrapper, self).__init__(model_prefix, name)
        self.data = json.load(open(dict_fname, 'r'))

    # check if user text is match to one of the keys
    def isMatch(self, text):
        for key, value in self.data.iteritems():
            if re.match(key, text, re.IGNORECASE):
                return True
        return False

    # return the key which matches
    def getMatch(self, text):
        for key, value in self.data.iteritems():
            if re.match(key, text, re.IGNORECASE):
                return key
        return False

    def get_response(self, user_id='', text='', context=None, **kwargs):
        logging.info('------------------------------------')
        logging.info('Generating dumb question for user %s.' % user_id)
        ctext = self._format_to_model(text, len(context))
        context.append(ctext)
        if self.isMatch(text):
            key = self.getMatch(text)
            response = random.choice(self.data[key])
        else:
            response = ''
        context.append(self._format_to_model(response, len(context)))
        return response, context


class DRQA_Wrapper(Model_Wrapper):
    """
    GIVE AN ANSWER RELATED TO THE ARTICLE
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(DRQA_Wrapper, self).__init__(model_prefix, name)
        self.articles = {}

    # check if user text is match to one of the keys
    def isMatch(self, text):
        for key, value in self.data.iteritems():
            if re.match(key, text, re.IGNORECASE):
                return True
        return False

    def preprocess(self, chat_id='', article_text='', **kwargs):
        logging.info("Saving the article for this chat state")
        self.articles[chat_id] = article_text

    # return the key which matches
    def getMatch(self, text):
        for key, value in self.data.iteritems():
            if re.match(key, text, re.IGNORECASE):
                return key
        return False

    def get_response(self, user_id='', text='', context='', article='', **kwargs):
        logging.info('------------------------------------')
        logging.info('Generating DRQA answer for user %s.' % user_id)
        ctext = self._format_to_model(text, len(context))
        context.append(ctext)
        response = ''
        article_present = False
        if user_id in self.articles:
            article_present = True
        if not isinstance(article,basestring):
            article = str(article)
        if article_present:
            if len(article) == 0:
                logging.info("DRQA taking saved article, only if present")
                article = self.articles[user_id]
            try:
                res = requests.post(DRQA_ENDURL+'/ask',
                                    json={'article': article, 'question': text})
                res_data = res.json()
                response = res_data['reply']['text']
            except Exception as e:
                print e
                logging.error(e)
            context.append(self._format_to_model(response, len(context)))
        return response, context


class NQG_Wrapper(Model_Wrapper):
    """
    GENERATES A QUESTION FOR EACH SENTENCE IN THE ARTICLE
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(NQG_Wrapper, self).__init__(model_prefix, name)
        self.questions = {}
        self.seen_user = []

    def preprocess(self, chat_id='', article_text='', **kwargs):
        # extract all sentences from the article
        logging.info('Preprocessing the questions for this article')
        # check condition if we use Spacy
        assert isinstance(article_text, basestring)
        # clean the text
        article_text = re.sub(
            r'^https?:\/\/.*[\r\n]*', '', article_text, flags=re.MULTILINE)
        sentences = sent_tokenize(article_text)
        try:
            res = requests.post(NQG_ENDURL, json={'sents': sentences})
            res_data = res.json()
            self.questions[chat_id] = res_data
            # remove duplicate questions
            all_preds = []
            rm_index = []
            for indx, item in enumerate(self.questions[chat_id]):
                item.update({'used':0})
                if item['pred'] not in all_preds:
                    all_preds.append(item['pred'])
                else:
                    rm_index.append(indx)
            logging.info('Preprocessed article')
            # pruning bad examples
            for i,preds in enumerate(self.questions[chat_id]):
                if 'source: source:' in preds['pred']:
                    rm_index.append(i)
            rm_index = list(set(rm_index))
            self.questions[chat_id] = [qs for i,qs in enumerate(self.questions[chat_id]) if i not in set(rm_index)]

            self.questions[chat_id].sort(key=lambda x:  x["score"])
        except Exception as e:
            logging.info('Error in NQG article fetching')
            logging.error(e)

    def get_response(self, user_id='', text='', context=None, article=None, **kwargs):
        logging.info('----------------------------------------')
        logging.info('Generating NQG question for user %s.' % user_id)
        logging.info('Context')
        logging.info(context)
        response = ''
        if len(self.questions) > 0 and user_id in self.questions:
            logging.info("Available questions : ")
            logging.info(self.questions[user_id])
            qs = [(i,q) for i,q in enumerate(self.questions[user_id]) if q['used'] == 0]
            if len(qs) > 0:
                response = qs[0][1]['pred']
                if user_id in self.seen_user:
                    self.questions[user_id][qs[0][0]]['used'] += 1
                else:
                    self.seen_user.append(user_id)
                self.questions[user_id].sort(key=lambda x: x["used"])

        context.append(self._format_to_model(response, len(context)))
        return response, context

    def clean(self, chat_id):
        del self.questions[chat_id]


class Echo_Wrapper(Model_Wrapper):
    """
    ECHO: RETURN THE INPUT
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(Echo_Wrapper, self).__init__(model_prefix, name)

    def get_response(self, user_id='', text='', context=None, article=None, **kwargs):
        logging.info('------------------------------------')
        logging.info('Generating Echo response for user %s.' % user_id)
        text = self._format_to_model(text, len(context))
        context.append(text)
        logging.info('Using context: %s' % ' '.join(list(context)))

        response = text
        response = self._format_to_user(response)
        context.append(self._format_to_model(response, len(context)))
        return response, context


class Topic_Wrapper(Model_Wrapper):
    """
    IF USER ASKS FOR THE TOPIC, RETURN TOPIC CLASSIFICATION USING FASTTEXT
    """

    def __init__(self, model_prefix, dict_fname, name, dir_name, model_name, top_k=2):
        super(Topic_Wrapper, self).__init__(model_prefix, name)
        # Read the classes once
        self.dir_name = dir_name
        self.model_name = model_name
        self.topics = []
        self.predicted = {}
        self.top_k = top_k
        self.query_string = 'cd {} && ./fasttext predict {} /tmp/{}_article.txt {} > /tmp/{}_preds.txt'
        with open(dir_name + 'classes.txt', 'r') as fp:
            for line in fp:
                self.topics.append(line.rstrip())

        self.topic_phrases = ["<topic>",
                              "This article is about <topic>",
                              "I think it's about <topic>",
                              "It's about <topic>",
                              "The article is related to <topic>"]

    def isMatch(self, text=''):
        # catch responses of the style "what is this article about"
        question_match_1 = ".*what\\s*(is|'?s|does)?\\s?(this|it|the)?\\s?(article)?\\s?(talks?)?\\s?(about)\\s*(\\?)*"
        # catch also responses of the style : "what do you think of this article"
        question_match_2 = ".*what\\sdo\\syou\\sthink\\s(of|about)\\s(this|it|the)?\\s?(article)\\s*\\?*"
        return re.match(question_match_1, text, re.IGNORECASE) or re.match(question_match_2, text, re.IGNORECASE)

    def preprocess(self, chat_id='', article_text='', **kwargs):
        """Calculate the article topic in preprocess call
        """
        logging.info("Started preprocesssing topics")
        # Write article to tmp file
        with codecs.open('/tmp/{}_article.txt'.format(chat_id), 'w', encoding='utf-8') as fp:
            fp.write(article_text.replace("\n", ""))
        # Use subprocess to call fasttext
        logging.info("Running fasttext ...")
        call_string = self.query_string.format(FASTTEXT_DIR,
                                               self.model_name, chat_id,
                                               self.top_k, chat_id)
        logging.info(call_string)
        outp = delegator.run(call_string)
        logging.info(outp.out)
        logging.info(outp.err)
        # store the topics in memory
        logging.info("Extracting predictions")
        self.predicted[chat_id] = []
        with open('/tmp/{}_preds.txt'.format(chat_id), 'r') as fp:
            for line in fp:
                p = line.split(' ')
                p = [int(pt.replace('__label__', '')) - 1 for pt in p]
                self.predicted[chat_id].append([self.topics[pt] for pt in p])
        assert len(self.predicted[chat_id]) == 1
        logging.info("Calculated topics for the article, which are {}".format(
            ','.join(self.predicted[chat_id][0])))

    def get_response(self, user_id='', text='', context=None, article=None, **kwargs):
        logging.info('---------------------------------')
        logging.info('Generating topic for the article')
        logging.info('Topics : {}'.format(self.predicted))
        logging.info('isMatch : {}'.format(self.isMatch(text)))
        if len(self.predicted) > 0 and self.isMatch(text):
            topic = self.predicted[user_id][0][0]  # give back the top topic
            topic_phrase_index = random.choice(range(len(self.topic_phrases)))
            response = self.topic_phrases[topic_phrase_index].replace(
                "<topic>", topic)
        else:
            response = ''
        context.append(self._format_to_model(response, len(context)))
        return response, context


class DrQA_Wiki_Wrapper(Model_Wrapper):
    """
    GIVE AN ANSWER RELATED TO WIKIPEDIA DUMP
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(DrQA_Wiki_Wrapper, self).__init__(model_prefix, name)

    def get_response(self, user_id='', text='', context=None, article=None, **kwargs):
        logging.info('------------------------------------')
        logging.info('Generating DrQA WIKI response for user %s.' % user_id)
        cont = zmq.Context()
        socket = cont.socket(zmq.REQ)
        socket.connect("ipc:///tmp/drqa.pipe")
        socket.send_json({'question': text, 'top_n': 1, 'n_docs': 5})
        reply = socket.recv_json()

        response = reply['span']
        response = self._format_to_user(response)
        context.append(self._format_to_model(response, len(context)))
        return response, context

# Shamelessly inspired from MILABOT


class FactGenerator_Wrapper(Model_Wrapper):
    """
    RETURN A FACT ACCRODING TO THE CONVERSATION HISTORY
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(FactGenerator_Wrapper, self).__init__(model_prefix, name)
        random_facts_path = '/root/convai/data/facts.txt'
        all_facts_embedding_path = '/root/convai/data/fact_embedding.mod'
        self.all_facts = codecs.open(
            random_facts_path, 'r', encoding='utf-8').readlines()
        self.all_facts = [all_fact.strip().replace(" .", ".").replace("\"", " ")
                          for all_fact in self.all_facts
                          if len(all_fact.strip().split()) > 2]

        self.fact_phrases = ["<fact>",
                             "Did you know that <fact>?",
                             "Do you know that <fact>?",
                             "Here's an interesting fact. <fact>",
                             "Here's a fact! <fact>"]

        self.wh_prefix_phrases = ["I'm not sure. However,",
                                  "I'm not sure. But.",
                                  "I'm not quite sure. But",
                                  "I don't have an answer for that. But",
                                  "I don't know. But."]

        self.w2v_path = '/root/convai/data/GoogleNews-vectors-negative300.bin'
        self.w2v = KeyedVectors.load_word2vec_format(
            self.w2v_path, binary=True)
        self.w2v_dim = self.w2v['hello'].shape[0]
        self.w2v_stopwords = conf.stopwords
        self.wh_words = conf.wh_words
        if os.path.exists(all_facts_embedding_path):
            self.all_facts_embeddings = pkl.load(
                open(all_facts_embedding_path, 'r'))
        else:
            self.all_facts_embeddings = np.zeros(
                (len(self.all_facts), self.w2v_dim), dtype='float32')
            for fact_index, fact in enumerate(self.all_facts):
                fact_embedding, fact_valid_embedding = self.get_utterance_embedding(
                    fact)
                self.all_facts_embeddings[fact_index, :] = fact_embedding
            pkl.dump(self.all_facts_embeddings, open(
                all_facts_embedding_path, 'w'))

    def get_utterance_embedding(self, utterance):
        tokens = utterance.replace("'s", "").replace("'t", "").replace("'d", "").replace("'ll", "").replace("'re", "").replace(
            "'ve", "").replace("'", " ").replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").replace("\"", " ").split()
        X = np.zeros((self.w2v_dim,))
        for tok in tokens:
            if (len(tok) > 1) and (tok not in self.w2v_stopwords):
                if tok in self.w2v:
                    X += self.w2v[tok]

        if np.linalg.norm(X) < 0.00000000001:
            return X, False
        else:
            return X / np.linalg.norm(X), True

    def get_response(self, user_id='', text='', context=None, **kwargs):
        context.append(text)
        if len(context) > 0:
            dialogue_history_flattened = ' '.join(context).lower()
            dialogue_history_embedding, is_valid_embedding = self.get_utterance_embedding(
                dialogue_history_flattened)
            last_user_utterance_lower = context[-1].replace("'", " ").replace("\"", " ").replace(
                ".", " ").replace("!", " ").replace("?", " ").replace(",", " ").lower().split()
            last_user_utterance_has_wh_word = False
            for word in last_user_utterance_lower:
                if word in self.wh_words:
                    last_user_utterance_has_wh_word = True
                    break
            scores = np.dot(self.all_facts_embeddings,
                            dialogue_history_embedding.T)
            facts_indices_sorted = scores.argsort()[::-1]
            for fact_index in facts_indices_sorted:
                fact = self.all_facts[fact_index]
                if fact.lower() not in dialogue_history_flattened:
                    fact_phrase_index = random.choice(
                        range(len(self.fact_phrases)))
                    fact_text = self.fact_phrases[fact_phrase_index].replace(
                        "<fact>", fact)
                    if last_user_utterance_has_wh_word:
                        fact_text = random.choice(
                            self.wh_prefix_phrases) + ' ' + fact_text
                    context.append(self._format_to_model(
                        fact_text, len(context)))
                    return fact_text, context


class AliceBot_Wrapper(Model_Wrapper):
    """
    USES ALICE_BOT TO GIVE A REPLY
    """

    def __init__(self, model_prefix, dict_fname, name):
        super(AliceBot_Wrapper, self).__init__(model_prefix, name)
        self.aliceBot = NLGAlice()

    def get_response(self, user_id='', text='', context=None, **kwargs):
        ctext = self._format_to_model(text, len(context))
        context.append(ctext)
        # strip context of special tokens
        clean_context = []
        if context and len(context) > 0:
            for cont in context:
                cln = re.sub("[\(\[\<].*?[\)\]\>]", "", cont)
                clean_context.append(cln)
        print clean_context
        response = ''
        try:
            response = self.aliceBot.compute_responses(clean_context, None)
            # prune response for presence of Alexa, Socialbot or MILA
            if 'Alexa' in response:
                response.replace('Alexa', 'Botty')
            if 'Socialbot' in response:
                response.replace('Socialbot', 'Convbot')
            if 'MILA' in response:
                response.replace('MILA', 'Hogwarts')
        except Exception as e:
            logging.error("Error generating alicebot response")
        context.append(self._format_to_model(response, len(context)))
        return response, context
