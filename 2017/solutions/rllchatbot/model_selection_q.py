import zmq
import sys
import config
import random
import spacy
import re
import time
import numpy as np
import emoji
import json
import cPickle as pkl
from datetime import datetime
from ranker import features
from ranker.estimators import Estimator, LONG_TERM_MODE, SHORT_TERM_MODE
from Queue import Queue
from threading import Thread
import multiprocessing
from multiprocessing import Pool, Process
import uuid
from models.wrapper import Dual_Encoder_Wrapper, Human_Imitator_Wrapper, HREDQA_Wrapper, CandidateQuestions_Wrapper, DumbQuestions_Wrapper, DRQA_Wrapper, NQG_Wrapper, Echo_Wrapper, Topic_Wrapper, FactGenerator_Wrapper, AliceBot_Wrapper
from models.wrapper import HRED_Wrapper
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)
conf = config.get_config()


# Script to select model conversation
# Initialize all the models here and then reply with the best answer
# ZMQ version. Here, all models are initialized in a separate thread
# Then the main process sends out next response and context as a PUB msg,
# which every model_client listens in SUB
# Then all models calculate the response and send it to parent using PUSH-PULL
# NN model selection algorithm can therefore work in the parent queue
# to calculate the score of all incoming msgs
# Set a hard time limit to discard the messages which are slow
# N.B. `import zmq` **has to be** the first import.


nlp = spacy.load('en')

# Utils


def mogrify(topic, msg):
    """ json encode the message and prepend the topic """
    return topic + ' ' + json.dumps(msg)


def demogrify(topicmsg):
    """ Inverse of mogrify() """
    json0 = topicmsg.find('{')
    topic = topicmsg[0:json0].strip()
    msg = json.loads(topicmsg[json0:])
    return topic, msg


class Policy:
    NONE = -1        # between each chat
    OPTIMAL = 1      # current set of rules
    # exploratory policies:
    HREDx2 = 2       # hred-reddit:0.5 & hred-twitter:0.5
    HREDx3 = 3       # hred-reddit:0.33 & hred-twitter:0.33 & hred-qa:0.33
    HREDx2_DE = 4    # hred-reddit:0.25 & hred-twitter:0.25 & DualEncoder:0.5
    HREDx2_DRQA = 5  # hred-reddit:0.25 & hred-twitter:0.25 & DrQA:0.5
    DE_DRQA = 6      # DualEncoder:0.5 & DrQA:0.5
    START = 7
    FIXED = 8        # when allowed_model = True
    LEARNED = 9      # Learned policy ranker
    BORED = 10


class ModelID:
    DRQA = 'drqa'
    DUAL_ENCODER = 'de'
    HUMAN_IMITATOR = 'de-human'
    HRED_REDDIT = 'hred-reddit'
    HRED_TWITTER = 'hred-twitter'
    DUMB_QA = 'dumb_qa'
    NQG = 'nqg'
    # FOLLOWUP_QA = 'followup_qa'
    CAND_QA = 'candidate_question'
    TOPIC = 'topic_model'
    FACT_GEN = 'fact_gen'
    ALICEBOT = 'alicebot'
    # ECHO = 'echo_model'  # just for debugging purposes
    ALL = 'all'          # stub to represent all allowable models


# TODO: make sure never used in other files before removing
# ALL_POLICIES = [Policy.OPTIMAL, Policy.HREDx2, Policy.HREDx3,
#                 Policy.HREDx2_DE, Policy.HREDx2_DRQA, Policy.DE_DRQA]

BORED_COUNT = 2

# wait time is the amount of time we wait to let the models respond.
# Instead of the previous architecture, now we would like to respond faster.
# So the focus is that even if some models are taking a lot of time,
# do not wait for them!
WAIT_TIME = 7

# PINGBACK
# Check every time if the time now - time last pinged back of a model
# is within PING_TIME. If not, revive
PING_TIME = 60

# IPC pipes
# Parent to models
COMMAND_PIPE = 'ipc:///tmp/command.pipe'
# models to parent
BUS_PIPE = 'ipc:///tmp/bus.pipe'
# parent to bot caller
PARENT_PIPE = 'ipc:///tmp/parent_push.pipe'
# bot to parent caller
PARENT_PULL_PIPE = 'ipc:///tmp/parent_pull.pipe'

###
# Load ranker models
###

# short term estimator
ranker_args_short = []
with open(conf.ranker['model_short'], 'rb') as fp:
    data_short, hidden_dims_short, hidden_dims_extra_short, activation_short, \
        optimizer_short, learning_rate_short, \
        model_path_short, model_id_short, model_name_short, _, _, _ \
        = pkl.load(fp)
# load the feature list used in short term ranker
feature_list_short = data_short[-1]
# reconstruct model_path just in case file have been moved:
model_path_short = conf.ranker['model_short'].split(model_id_short)[0]
if model_path_short.endswith('/'):
    model_path_short = model_path_short[:-1]  # ignore the last '/'
logging.info(model_path_short)

# long term estimator
ranker_args_long = []
with open(conf.ranker['model_long'], 'rb') as fp:
    data_long, hidden_dims_long, hidden_dims_extra_long, activation_long, \
        optimizer_long, learning_rate_long, \
        model_path_long, model_id_long, model_name_long, _, _, _ \
        = pkl.load(fp)
# load the feature list user in short term ranker
feature_list_long = data_long[-1]
# reconstruct model_path just in case file have been moved:
model_path_long = conf.ranker['model_long'].split(model_id_long)[0]
if model_path_long.endswith('/'):
    model_path_long = model_path_long[:-1]  # ignore the last '/'
logging.info(model_path_long)

assert feature_list_short == feature_list_long
# NOTE: could also check equality between the following variables, but its redundant:
# - hidden_dims_short       & hidden_dims_long
# - hidden_dims_extra_short & hidden_dims_extra_long
# - activation_short        & activation_long
# - optimizer_short         & optimizer_long
# - learning_rate_short     & learning_rate_long

logging.info("creating ranker feature instances...")
start_creation_time = time.time()
feature_objects, feature_dim = features.initialize_features(feature_list_short)
logging.info("created all feature instances in %s sec" % (time.time() - start_creation_time))


class ModelClient(multiprocessing.Process):
    """
    Client Process for individual models. Initialize the model
    and subscribe to channel to listen for updates
    """

    def __init__(self, task_queue, result_queue, model_name, estimate=True):
        multiprocessing.Process.__init__(self)
        self.model_name = model_name
        self.estimate = estimate
        self.task_queue = task_queue
        self.result_queue = result_queue

    def build(self):
        # select and initialize models
        if self.model_name == ModelID.HRED_REDDIT:
            logging.info("Initializing HRED Reddit")
            self.model = HRED_Wrapper(conf.hred['reddit_model_prefix'],
                                      conf.hred['reddit_dict_file'],
                                      ModelID.HRED_REDDIT)
            self.estimate = True  # always sampled according to score
        if self.model_name == ModelID.HRED_TWITTER:
            logging.info("Initializing HRED Twitter")
            self.model = HRED_Wrapper(conf.hred['twitter_model_prefix'],
                                      conf.hred['twitter_dict_file'],
                                      ModelID.HRED_TWITTER)
            self.estimate = True  # always sampled according to score
        # if model_name == ModelID.FOLLOWUP_QA:
        #     logging.info("Initializing HRED Followup")
        #     self.model = HREDQA_Wrapper(conf.followup['model_prefix'],
        #                                 conf.followup['dict_file'],
        #                                 ModelID.FOLLOWUP_QA)
        #     self.estimate = True  # sampled according to score when user didn't ask a question
        if self.model_name == ModelID.DUAL_ENCODER:
            logging.info("Initializing Dual Encoder")
            self.model = Dual_Encoder_Wrapper(conf.de['reddit_model_prefix'],
                                              conf.de['reddit_data_file'],
                                              conf.de['reddit_dict_file'],
                                              ModelID.DUAL_ENCODER)
            self.estimate = True  # always sampled according to score
        if self.model_name == ModelID.HUMAN_IMITATOR:
            logging.info("Initializing Dual Encoder on Human data")
            self.model = Human_Imitator_Wrapper(conf.de['convai-h2h_model_prefix'],
                                                conf.de['convai-h2h_data_file'],
                                                conf.de['convai-h2h_dict_file'],
                                                ModelID.HUMAN_IMITATOR)
            self.estimate = True  # sampled according to score when user is bored
        if self.model_name == ModelID.DRQA:
            logging.info("Initializing DRQA")
            self.model = DRQA_Wrapper('', '', ModelID.DRQA)
        if self.model_name == ModelID.DUMB_QA:
            logging.info("Initializing DUMB QA")
            self.model = DumbQuestions_Wrapper(
                '', conf.dumb['dict_file'], ModelID.DUMB_QA)
            self.estimate = False  # only used when user typed a simple enough turn
        if self.model_name == ModelID.NQG:
            logging.info("Initializing NQG")
            self.model = NQG_Wrapper('', '', ModelID.NQG)
            # sampled according to score when user is bored or when user didn't ask a question
            self.estimate = True
        # if model_name == ModelID.ECHO:
        #     logging.info("Initializing Echo")
        #     self.model = Echo_Wrapper('', '', ModelID.ECHO)
        #     self.estimate = False
        if self.model_name == ModelID.CAND_QA:
            logging.info("Initializing Candidate Questions")
            self.model = CandidateQuestions_Wrapper('',
                                                    conf.candidate['dict_file'],
                                                    ModelID.CAND_QA)
            # sampled according to score when user is bored or when user didn't ask a question
            self.estimate = True
        if self.model_name == ModelID.TOPIC:
            logging.info("Initializing topic model")
            self.model = Topic_Wrapper('', '', '', conf.topic['dir_name'],
                                       conf.topic['model_name'], conf.topic['top_k'])
            self.estimate = False  # only used when user requested article topic
        if self.model_name == ModelID.FACT_GEN:
            logging.info("Initializing fact generator")
            self.model = FactGenerator_Wrapper('', '', '')
            self.estimate = True  # sampled according to score when user is bored
        if self.model_name == ModelID.ALICEBOT:
            logging.info("Initializing Alicebot")
            self.model = AliceBot_Wrapper('', '', '')
            self.estimate = True  # always sampled according to score 

        self.is_running = True
        import tensorflow as tf
        logging.info("Building NN Ranker")
        ###
        # Build NN ranker models and set their parameters
        ###
        # NOTE: stupid theano can't load two graph within the same session -_-
        # SOLUTION: taken from https://stackoverflow.com/a/41646443
        self.model_graph_short = tf.Graph()
        with self.model_graph_short.as_default():
            self.estimator_short = Estimator(
                data_short, hidden_dims_short, hidden_dims_extra_short, activation_short,
                optimizer_short, learning_rate_short,
                model_path_short, model_id_short, model_name_short
            )
        self.model_graph_long = tf.Graph()
        with self.model_graph_long.as_default():
            self.estimator_long = Estimator(
                data_long, hidden_dims_long, hidden_dims_extra_long, activation_long,
                optimizer_long, learning_rate_long,
                model_path_long, model_id_long, model_name_long
            )
        logging.info("Init tf session")
        # Wrap ANY call to the rankers within those two sessions:
        self.sess_short = tf.Session(graph=self.model_graph_short)
        self.sess_long = tf.Session(graph=self.model_graph_long)
        logging.info("Done init tf session")
        # example: reload trained parameters
        logging.info("Loading trained params short-term estimator")
        with self.sess_short.as_default():
            with self.model_graph_short.as_default():
                self.estimator_short.load(
                    self.sess_short, model_path_short, model_id_short, model_name_short)
        logging.info("Loading trained params long-term estimator")
        with self.sess_long.as_default():
            with self.model_graph_long.as_default():
                self.estimator_long.load(
                    self.sess_long, model_path_long, model_id_long, model_name_long)

        logging.info("Done building NN ranker")

        self.warmup()

    def warmup(self):
        """ Warm start the models before execution """
        if self.model_name != ModelID.DRQA:
            _, _ = self.model.get_response(1, 'test_statement', [])
        else:
            _, _ = self.model.get_response(1, 'Where is Daniel?', [], nlp(
                unicode('Daniel went to the kitchen')))
        self.result_queue.put(None)

    def run(self):
        """ Main running point of the process
        """
        # building and warming up
        start_time_build = time.time()
        self.build()
        logging.info("Built model {}".format(self.model_name))

        logging.info("Model {} listening for requests".format(self.model_name))
        while(True):
            msg = self.task_queue.get()
            msg['user_id'] = msg['chat_id']
            response = ''
            if 'control' in msg and msg['control'] == 'preprocess':
                self.model.preprocess(**msg)
            else:
                response, context = self.model.get_response(**msg)

            # if blank response, do not push it in the channel
            if len(response) > 0:
                if len(context) > 0 and self.estimate:
                    # calculate NN estimation
                    logging.info(
                        "Start feature calculation for model {}".format(self.model_name))
                    raw_features = features.get(
                        feature_objects,
                        feature_dim,
                        msg['article_text'],
                        msg['all_context'] + [context[-1]],
                        response
                    )
                    logging.info(
                        "Done feature calculation for model {}".format(self.model_name))
                    # Run approximator and save the score in packet
                    logging.info(
                        "Scoring the candidate response for model {}".format(self.model_name))
                    # reshape raw_features to fit the ranker format
                    assert len(raw_features) == feature_dim
                    candidate_vector = raw_features.reshape(
                        1, feature_dim)  # make an array of shape (1, input)
                    # Get predictions for this candidate response:
                    with self.sess_short.as_default():
                        with self.model_graph_short.as_default():
                            logging.info("estimator short predicting")
                            # get predicted class (0: downvote, 1: upvote), and confidence (ie: proba of upvote)
                            vote, conf = self.estimator_short.predict(
                                SHORT_TERM_MODE, candidate_vector)
                            # sanity check with batch size of 1
                            assert len(vote) == len(conf) == 1
                    with self.sess_long.as_default():
                        with self.model_graph_long.as_default():
                            logging.info("estimator long prediction")
                            # get the predicted end-of-dialogue score:
                            pred, _ = self.estimator_long.predict(
                                LONG_TERM_MODE, candidate_vector)
                            # sanity check with batch size of 1
                            assert len(pred) == 1
                    vote = vote[0]  # 0 = downvote ; 1 = upvote
                    conf = conf[0]  # 0.0 < Pr(upvote) < 1.0
                    score = pred[0]  # 1.0 < end-of-chat score < 5.0
                else:
                    vote = -1
                    conf = 0
                    score = -1

                resp_msg = {'text': response, 'context': context,
                            'model_name': self.model_name,
                            'chat_id': msg['chat_id'],
                            'chat_unique_id': msg['chat_unique_id'],
                            'vote': str(vote),
                            'conf': str(conf),
                            'score': str(score)}
                self.result_queue.put(resp_msg)
            else:
                self.result_queue.put({})
        return

## ResponseQuerier

class ResponseModelsQuerier(object):

    def __init__(self, modelIDs):
        self.modelIDs = modelIDs
        self.models = []
        for model_name in self.modelIDs:
            tasks = multiprocessing.JoinableQueue(1)
            results = multiprocessing.Queue(1)

            model_runner = ModelClient(tasks, results, model_name)
            model_runner.daemon = True
            model_runner.start()

            self.models += [{"model_runner": model_runner, "tasks": tasks, 
                "results": results, "model_name":model_name}]


        # Make sure that all models are started
        for model in self.models:
            model_name = model['model_name']
            try:
                response = model["results"].get(timeout=90)  # Waiting 10 minutes for each model to initialize.
            except Exception as e:
                raise RuntimeError("{} took too long to build.".format(model_name))

            if isinstance(response, Exception):
                print("\n{} Failed to initialize with error ({}). See logs in ./logs/models/".format(model_name, response))
                exit(1)

    def get_response(self, msg):
        for model in self.models:
            if 'query_models' in msg:
                if model['model_name'] in msg['query_models']:
                    model["tasks"].put(msg)
            else:
                model["tasks"].put(msg)

        candidate_responses = {}
        for model in self.models:
            if 'query_models' in msg:
                if model['model_name'] in msg['query_models']:
                    responses = model["results"].get()
                else:
                    responses = {}
            else:
                responses = model["results"].get()

            if isinstance(responses, Exception):
                model_name = model['model_name']
                logging.error("\n{0} failed to compute response with error ({1}). \n{0} has been removed from running models.".format(model_name, responses))
                self.models.remove(model)

                if len(self.models) == 0:
                    print("All models failed. Exiting ...")
                    exit(1)
            else:
                if len(responses) > 0:
                    candidate_responses[model['model_name']] = responses
        return candidate_responses



class ModelSelectionAgent(object):
    def __init__(self):
        self.article_text = {}     # map from chat_id to article text
        self.chat_history = {}     # save the context / response pairs for a particular chat here
        self.candidate_model = {}  # map from chat_id to a simple model for the article
        self.article_nouns = {}    # map from chat_id to a list of nouns in the article
        self.boring_count = {}     # number of times the user responded with short answer
        self.used_models = {}

        # list of generic words to detect if user is bored
        self.generic_words_list = []
        with open('/root/convai/data/generic_list.txt') as fp:
            for line in fp:
                self.generic_words_list.append(line.strip())
        self.generic_words_list = set(self.generic_words_list)  # remove duplicates

        self.modelIds = [
            # ModelID.ECHO,          # return user input
            ModelID.CAND_QA,       # return a question about an entity in the article
            ModelID.HRED_TWITTER,  # general generative model on twitter data
            ModelID.HRED_REDDIT,   # general generative model on reddit data
            # ModelID.FOLLOWUP_QA,   # general generative model on questions (ie: what? why? how?)
            ModelID.DUMB_QA,       # return predefined answer to 'simple' questions
            ModelID.DRQA,          # return answer about the article
            ModelID.NQG,           # generate a question for each sentence in the article
            ModelID.TOPIC,         # return article topic
            ModelID.FACT_GEN,      # return a fact based on conversation history
            ModelID.ALICEBOT,      # give all responsabilities to A.L.I.C.E. ...
            # ModelID.DUAL_ENCODER,  # return a reddit turn
            # ModelID.HUMAN_IMITATOR  # return a human turn from convai round1
        ]

        self.response_models = ResponseModelsQuerier(self.modelIds)

    def get_response(self, chat_id, text, context, allowed_model=None, control=None):
        # create a chat_id + unique ID candidate responses field
        # chat_unique_id is needed to uniquely determine the return
        # for each call
        chat_unique_id = str(chat_id) + '_' + str(uuid.uuid4())
        is_start = False

        logging.info(chat_id)
        logging.info(self.article_nouns)

        # if text contains /start, don't add it to the context
        if '/start' in text:
            is_start = True
            # remove start token
            text = re.sub(r'\/start', '', text)
            # remove urls
            text = re.sub(r'https?:\/\/.*[\r\n]*',
                          '', text, flags=re.MULTILINE)
            # save the article for later use
            self.article_text[chat_id] = text
            article_nlp = nlp(unicode(text))
            # save all nouns from the article
            self.article_nouns[chat_id] = [
                p.lemma_ for p in article_nlp if p.pos_ in ['NOUN', 'PROPN']
            ]

            # initialize bored count to 0 for this new chat
            self.boring_count[chat_id] = 0

            # initialize chat history
            self.chat_history[chat_id] = []

            # initialize model usage history
            self.used_models[chat_id] = []

            # preprocessing
            logging.info("Preprocessing call")
            _ = self.response_models.get_response({
                'control':'preprocess',
                'article_text':self.article_text[chat_id],
                'chat_id':chat_id,
                'chat_unique_id': chat_unique_id
            })
            logging.info("preprocessed")

            # cand and nqg
            query_models = [ModelID.NQG, ModelID.CAND_QA]

            logging.info("fire call")
            candidate_responses = self.response_models.get_response({
                'query_models': query_models,
                'article_text':self.article_text[chat_id],
                'chat_id': chat_id,
                'chat_unique_id': chat_unique_id,
                'text':'',
                'context':context,
                'all_context':self.chat_history[chat_id]
            })
            logging.info("received response")

        else:
            # fire global query
            # making sure dict initialized
            if chat_id not in self.boring_count:
                self.boring_count[chat_id] = 0

            if chat_id not in self.chat_history:
                self.chat_history[chat_id] = []

            if chat_id not in self.used_models:
                self.used_models[chat_id] = []

            article_text = ''
            all_context = []
            if chat_id in self.article_text:
                article_text = self.article_text[chat_id]
            if chat_id in self.chat_history:
                all_context = self.chat_history[chat_id]

            candidate_responses = self.response_models.get_response({
                    'article_text':article_text,
                    'chat_id': chat_id,
                    'chat_unique_id': chat_unique_id,
                    'text':text,
                    'context':context,
                    'all_context':all_context
                })

        # got the responses, now choose which one to send.
        response = None
        if is_start:
            # TODO: replace this with a proper choice / always NQG?
            choices = list(set([ModelID.CAND_QA, ModelID.NQG])
                           .intersection(set(candidate_responses.keys())))
            if len(choices) > 0:
                selection = random.choice(choices)
                response = candidate_responses[selection]
                response['policyID'] = Policy.START
        else: 
            logging.info("Removing duplicates")
            candidate_responses = self.no_duplicate(chat_id, candidate_responses)
            # if text contains emoji's, strip them
            #text, emojis = strip_emojis(text)
            # check if the text contains wh words
            ntext = nlp(unicode(text))
            nt_words = [p.lemma_ for p in ntext]
            has_wh_word = False
            for word in nt_words:
                if word in set(conf.wh_words):
                    has_wh_word = True
                    break
            #if emojis and len(text.strip()) < 1:
            #    # if text had only emoji, give back the emoji itself
            #    response = {'response': emojis, 'context': context,
            #                'model_name': 'emoji', 'policy': Policy.NONE}
            # if query falls under dumb questions, respond appropriately
            if ModelID.DUMB_QA in candidate_responses:
                logging.info("Matched dumb preset patterns")
                response = candidate_responses[ModelID.DUMB_QA]
                response['policyID'] = Policy.FIXED
            # if query falls under topic request, respond with the article topic
            elif ModelID.TOPIC in candidate_responses:
                logging.info("Matched topic preset patterns")
                response = candidate_responses[ModelID.TOPIC]
                response['policyID'] = Policy.FIXED
            # if query is a question, try to reply with DrQA
            elif has_wh_word or ("which" in set(nt_words)
                                 and "?" in set(nt_words)):
                # get list of common nouns between article and question
                if chat_id in self.article_nouns:
                    logging.info(self.article_nouns[chat_id])
                    common = list(set(self.article_nouns[chat_id]).intersection(
                        set(nt_words)))
                else:
                    logging.info("no article nouns saved")
                    common = []
                logging.info(
                    'Common nouns between question and article: {}'.format(common))
                # if there is a common noun between question and article
                # select DrQA
                if len(common) > 0 and ModelID.DRQA in candidate_responses:
                    response = candidate_responses[ModelID.DRQA]
                    response['policyID'] = Policy.FIXED

        if not response:
            # remove duplicates responses from k nearest chats
            # no_duplicate(chat_id, chat_unique_id)
            # Ranker based selection
            best_model, dont_consider = self.ranker(candidate_responses)
            if dont_consider and len(dont_consider) > 0:
                for model, score in dont_consider:
                    # remove the models from futher consideration
                    del candidate_responses[model]

            # Reduce confidence of CAND_QA
            if ModelID.CAND_QA in candidate_responses:
                cres = candidate_responses[ModelID.CAND_QA]
                cres_conf = float(cres['conf'])
                cres['conf'] = str(cres_conf / 2) # half the confidence
                candidate_responses[ModelID.CAND_QA] = cres

            # Bored model selection (TODO: nlp() might be taking time)
            nt_sent = nlp(unicode(text))
            nt_words = [p.lemma_ for p in nt_sent]
            # check if user said only generic words:
            generic_turn = True
            for word in nt_words:
                if word not in self.generic_words_list:
                    generic_turn = False
                    break
            # if text contains 2 words or less, add 1 to the bored count
            # also consider the case when the user says only generic things
            if len(text.strip().split()) <= 2 or generic_turn:
                self.boring_count[chat_id] += 1
            # list of available models to use if bored
            bored_models = [ModelID.NQG, ModelID.FACT_GEN,
                            ModelID.CAND_QA, ModelID.HUMAN_IMITATOR]
            boring_avl = list(
                set(candidate_responses.keys()).intersection(set(bored_models)))
            # if user is bored, change the topic by asking a question
            # (only if that question is not asked before)
            if self.boring_count[chat_id] >= BORED_COUNT and len(boring_avl) > 0:
                # assign model selection probability based on estimator confidence
                confs = [float(candidate_responses[model]['conf'])
                         for model in boring_avl]
                norm_confs = confs / np.sum(confs)
                selection = np.random.choice(boring_avl, 1, p=norm_confs)[0]
                response = candidate_responses[selection]
                response['policyID'] = Policy.BORED
                self.boring_count[chat_id] = 0  # reset bored count to 0

            # If not bored, then select from best model
            elif best_model:
                response = candidate_responses[best_model]
                response['policyID'] = Policy.LEARNED
            else:
                # Sample from the other models based on confidence probability
                # TODO: have choice of probability. Given the past model usage,
                # if HRED_TWITTER or HRED_REDDIT is used, then decay the
                # probability by small amount
                # randomly decide a model to query to get a response:
                models = [ModelID.HRED_REDDIT, ModelID.HRED_TWITTER,
                          ModelID.DUAL_ENCODER, ModelID.ALICEBOT]
                has_wh_word = False
                for word in nt_words:
                    if word in set(conf.wh_words):
                        has_wh_word = True
                        break
                if has_wh_word or ("which" in set(nt_words)
                                   and "?" in set(nt_words)):
                    # if the user asked a question, also consider DrQA
                    models.append(ModelID.DRQA)
                else:
                    # if the user didn't ask a question, also consider models
                    # that ask questions: hred-qa, nqg, and cand_qa
                    models.extend([ModelID.NQG, ModelID.CAND_QA])

                available_models = list(
                    set(candidate_responses.keys()).intersection(models))
                if len(available_models) > 0:
                    # assign model selection probability based on estimator confidence
                    confs = [float(candidate_responses[model]['conf'])
                             for model in available_models]
                    norm_confs = confs / np.sum(confs)
                    chosen_model = np.random.choice(
                        available_models, 1, p=norm_confs)
                    response = candidate_responses[chosen_model[0]]
                    response['policyID'] = Policy.OPTIMAL

        # if still no response, then just send a random fact
        if not response or 'text' not in response:
            logging.warn("Failure to obtain a response, using fact gen")
            response = candidate_responses[ModelID.FACT_GEN]
            response['policyID'] = Policy.FIXED

        # Now we have a response, so send it back to bot host
        # add user and response pair in chat_history
        self.chat_history[response['chat_id']].append(response['context'][-1])
        self.chat_history[response['chat_id']].append(response['text'])
        self.used_models[chat_id].append(response['model_name'])
        # Again use ZMQ, because lulz
        response['control'] = control
        logging.info("Done selecting best model")

        return response



    # given a set of model_responses, rank the best one based on
    # the following policy:
    # return ModelID of the model to select, else None
    # also return a list of models to NOT consider, else None
    # which indicates to take the pre-calculated policy


    def ranker(self, candidate_responses):
        # array containing tuple of (model_name, rank_score) for 1
        consider_models = []
        dont_consider_models = []  # for 0
        all_models = []  # for debugging purpose
        always_consider = [ModelID.HRED_REDDIT, ModelID.HRED_TWITTER,
                           ModelID.DUAL_ENCODER, ModelID.ALICEBOT]
        logging.info("Ranking among models")
        for model, response in candidate_responses.iteritems():
            conf = float(response['conf'])
            vote = float(response['vote'])
            score = float(response['score'])
            logging.info("{} - {} - {}".format(model, conf, score))
            if conf > 0.75:
                rank_score = conf * score
                # only consider these models
                if model in always_consider:
                    consider_models.append((model, rank_score))
            if conf < 0.25 and conf != 0:
                rank_score = conf * score
                if model != ModelID.FACT_GEN:  # keep fact generator model for failure handling case
                    dont_consider_models.append((model, rank_score))
            all_models.append((model, conf * score))
        all_models = sorted(all_models, key=lambda x: x[1], reverse=True)
        logging.info(all_models)
        consider = None
        dont_consider = None
        if len(consider_models) > 0:
            consider_models = sorted(
                consider_models, key=lambda x: x[1], reverse=True)
            consider = consider_models[0][0]
        elif len(dont_consider_models) > 0:
            dont_consider = dont_consider_models
        return consider, dont_consider

    def no_duplicate(self, chat_id, candidate_responses, k=5):
        del_models = []
        for model, response in candidate_responses.iteritems():
            if chat_id in self.chat_history and response['text'] in set(self.chat_history[chat_id][-k:]):
                del_models.append(model)
        for dm in del_models:
            del candidate_responses[dm]
        return candidate_responses


if __name__ == '__main__':
    model_selection_agent = ModelSelectionAgent()

    logging.info("====================================")
    logging.info("======RLLCHatBot Active=============")
    logging.info("All modules of the bot have been loaded.")
    logging.info("Thanks for your patience")
    logging.info("-------------------------------------")
    logging.info("Made with <3 in Montreal")
    logging.info("Reasoning & Learning Lab, McGill University")
    logging.info("Fall 2017")
    logging.info("=====================================")

    while True:
        user_utt = raw_input("USER : ")
        response = model_selection_agent.get_response(111, user_utt, [])
        print response
