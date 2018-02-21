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


nlp = spacy.load('en', parser=False, entity=False)

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


class ModelClient():
    """
    Client Process for individual models. Initialize the model
    and subscribe to channel to listen for updates
    """

    def __init__(self, model_name, estimate=True):
        # Process.__init__(self)
        self.model_name = model_name
        self.estimate = estimate
        # select and initialize models
        if model_name == ModelID.HRED_REDDIT:
            logging.info("Initializing HRED Reddit")
            self.model = HRED_Wrapper(conf.hred['reddit_model_prefix'],
                                      conf.hred['reddit_dict_file'],
                                      ModelID.HRED_REDDIT)
            self.estimate = True  # always sampled according to score
        if model_name == ModelID.HRED_TWITTER:
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
        if model_name == ModelID.DUAL_ENCODER:
            logging.info("Initializing Dual Encoder")
            self.model = Dual_Encoder_Wrapper(conf.de['reddit_model_prefix'],
                                              conf.de['reddit_data_file'],
                                              conf.de['reddit_dict_file'],
                                              ModelID.DUAL_ENCODER)
            self.estimate = True  # always sampled according to score
        if model_name == ModelID.HUMAN_IMITATOR:
            logging.info("Initializing Dual Encoder on Human data")
            self.model = Human_Imitator_Wrapper(conf.de['convai-h2h_model_prefix'],
                                                conf.de['convai-h2h_data_file'],
                                                conf.de['convai-h2h_dict_file'],
                                                ModelID.HUMAN_IMITATOR)
            self.estimate = True  # sampled according to score when user is bored
        if model_name == ModelID.DRQA:
            logging.info("Initializing DRQA")
            self.model = DRQA_Wrapper('', '', ModelID.DRQA)
        if model_name == ModelID.DUMB_QA:
            logging.info("Initializing DUMB QA")
            self.model = DumbQuestions_Wrapper(
                '', conf.dumb['dict_file'], ModelID.DUMB_QA)
            self.estimate = False  # only used when user typed a simple enough turn
        if model_name == ModelID.NQG:
            logging.info("Initializing NQG")
            self.model = NQG_Wrapper('', '', ModelID.NQG)
            # sampled according to score when user is bored or when user didn't ask a question
            self.estimate = True
        # if model_name == ModelID.ECHO:
        #     logging.info("Initializing Echo")
        #     self.model = Echo_Wrapper('', '', ModelID.ECHO)
        #     self.estimate = False
        if model_name == ModelID.CAND_QA:
            logging.info("Initializing Candidate Questions")
            self.model = CandidateQuestions_Wrapper('',
                                                    conf.candidate['dict_file'],
                                                    ModelID.CAND_QA)
            # sampled according to score when user is bored or when user didn't ask a question
            self.estimate = True
        if model_name == ModelID.TOPIC:
            logging.info("Initializing topic model")
            self.model = Topic_Wrapper('', '', '', conf.topic['dir_name'],
                                       conf.topic['model_name'], conf.topic['top_k'])
            self.estimate = False  # only used when user requested article topic
        if model_name == ModelID.FACT_GEN:
            logging.info("Initializing fact generator")
            self.model = FactGenerator_Wrapper('', '', '')
            self.estimate = True  # sampled according to score when user is bored
        if model_name == ModelID.ALICEBOT:
            logging.info("Initializing Alicebot")
            self.model = AliceBot_Wrapper('', '', '')
            self.estimate = True  # always sampled according to score
        # message queue. This contains the responses generated by the model
        self.queue = Queue()
        # process queue. This contains the responses to be processed
        self.process_queue = Queue()
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
        logging.info("Init session")
        # Wrap ANY call to the rankers within those two sessions:
        self.sess_short = tf.Session(graph=self.model_graph_short)
        self.sess_long = tf.Session(graph=self.model_graph_long)
        logging.info("Done init session")
        # example: reload trained parameters
        logging.info("Loading trained params short")
        with self.sess_short.as_default():
            with self.model_graph_short.as_default():
                self.estimator_short.load(
                    self.sess_short, model_path_short, model_id_short, model_name_short)
        logging.info("Loading trained params long")
        with self.sess_long.as_default():
            with self.model_graph_long.as_default():
                self.estimator_long.load(
                    self.sess_long, model_path_long, model_id_long, model_name_long)

        logging.info("Done building NN")
        self.discard_list = set()
        self.done_process = set()

        self.warmup()
        self.run()

    def warmup(self):
        """ Warm start the models before execution """
        if self.model_name != ModelID.DRQA:
            _, _ = self.model.get_response(1, 'test_statement', [])
        else:
            _, _ = self.model.get_response(1, 'Where is Daniel?', [], nlp(
                unicode('Daniel went to the kitchen')))

    def respond(self):
        """ Reply to the master on PUSH channel with the responses generated
        """
        socket = self.producer_context.socket(zmq.PUSH)
        socket.connect(BUS_PIPE)
        logging.info("Model {} push channel active".format(self.model_name))
        while self.is_running:
            msg = self.queue.get()
            if 'control' in msg and msg['control'] == 'ack':
                pass
            else:
                logging.info("Sending back info")
                logging.info(msg)
            socket.send_json(msg)
            self.queue.task_done()

    def act(self):
        """subscribe to master messages, and process them
        If msg contains key "control", process and exit
        """
        socket_b = self.consumer_context.socket(zmq.SUB)
        socket_b.connect(COMMAND_PIPE)
        socket_b.setsockopt(zmq.SUBSCRIBE, "user_response")
        # also subscribe to self topic
        socket_b.setsockopt(zmq.SUBSCRIBE, self.model_name)
        logging.info("Model {} subscribed to channels".format(self.model_name))
        while self.is_running:
            packet = socket_b.recv()
            topic, msg = demogrify(packet)
            if 'control' in msg:
                if msg['control'] == 'init':
                    logging.info(
                        "Model {} received init".format(self.model_name))
                if msg['control'] == 'preprocess':
                    if 'chat_id' in msg:
                        msg['user_id'] = msg['chat_id']
                    self.model.preprocess(**msg)
                if msg['control'] == 'discard':
                    if msg['chat_unique_id'] in self.done_process:
                        self.done_process.discard(msg['chat_unique_id'])
                    else:
                        self.discard_list.add(msg['chat_unique_id'])
                if msg['control'] == 'exit':
                    logging.info("Received exit signal, model {}"
                                 .format(self.model_name))
                    self.is_running = False

            else:
                try:
                    self.process_queue.put({'msg': msg})

                except Exception as e:
                    logging.error(e)
                    # shutdown process
                    self.shutdown()

    def process(self):
        """ Process the incoming msg and get a response
        """
        while(self.is_running):
            proc_msg = self.process_queue.get()
            msg = proc_msg['msg']
            if 'chat_unique_id' not in msg:
                self.process_queue.task_done()
                continue
            elif  msg['chat_unique_id'] not in self.discard_list:
                if 'chat_id' in msg:
                    msg['user_id'] = msg['chat_id']
                    response, context = self.model.get_response(**msg)

                # if blank response, do not push it in the channel
                if len(response) > 0:
                    if len(context) > 0 and self.estimate:
                        # calculate NN estimation
                        logging.info(
                            "Start feature calculation for model {}".format(self.model_name))
                        feat = features.get(
                            msg['article_text'], msg['all_context'] +
                            [context[-1]],
                            response, feature_list_short)
                        # recall: `feature_list_short` & `feature_list_long` are the same
                        logging.info(
                            "Done feature calculation for model {}".format(self.model_name))
                        # Run approximator and save the score in packet
                        logging.info(
                            "Scoring the candidate response for model {}".format(self.model_name))
                        # Get the input vector to the estimators from the feature instances lise:
                        candidate_vector = np.concatenate(
                            [feat[idx].feat for idx in range(len(feature_list_short))])
                        input_dim = len(candidate_vector)
                        candidate_vector = candidate_vector.reshape(
                            1, input_dim)  # make an array of shape (1, input)
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
                    self.queue.put(resp_msg)
                    self.done_process.add(msg['chat_unique_id'])
            else:
                # discard
                self.discard_list.discard(msg['chat_unique_id'])
            self.process_queue.task_done()




    def run(self):
        """Fire off the client"""
        self.producer_context = zmq.Context()
        self.consumer_context = zmq.Context()
        try:
            logging.info(
                "Starting {} response channel".format(self.model_name))
            resp_thread = Thread(target=self.respond)
            resp_thread.daemon = True
            resp_thread.start()
            logging.info("Starting {} act channel".format(self.model_name))
            act_thread = Thread(target=self.act)
            act_thread.daemon = True
            act_thread.start()
            logging.info("Starting {} process thread".format(self.model_name))
            process_thread = Thread(target=self.process)
            process_thread.daemon = True
            process_thread.start()
            while self.is_running:
                # Ping back to let parent know its alive
                self.queue.put(
                    {'control': 'ack', 'model_name': self.model_name})
                time.sleep(10)
            logging.info("Exiting {} client".format(self.model_name))
        except (KeyboardInterrupt, SystemExit):
            self.shutdown()

    def shutdown(self):
        """Clean shutdown process"""
        logging.info("Shutting down {} client".format(self.model_name))
        self.sess_short.close()
        self.sess_long.close()
        sys.exit(0)


# Initialize variables

article_text = {}     # map from chat_id to article text
chat_history = {}     # save the context / response pairs for a particular chat here
candidate_model = {}  # map from chat_id to a simple model for the article
article_nouns = {}    # map from chat_id to a list of nouns in the article
boring_count = {}     # number of times the user responded with short answer
# policy_mode = Policy.NONE  # TODO: make sure never used before removing
job_queue = Queue()
response_queue = Queue()
model_responses = {}
# This dictionary should contain an array per chat_id on the history of used models
used_models = {}

# list of generic words to detect if user is bored
generic_words_list = []
with open('/root/convai/data/generic_list.txt') as fp:
    for line in fp:
        generic_words_list.append(line.strip())
generic_words_list = set(generic_words_list)  # remove duplicates

# self.models = [self.hred_twitter, self.hred_reddit,
#               self.dual_enc, self.qa_hred, self.dumb_qa, self.drqa]
# modelIds = [ModelID.HRED_TWITTER, ModelID.HRED_REDDIT, ModelID.DUAL_ENCODER,
#                 ModelID.FOLLOWUP_QA, ModelID.DUMB_QA, ModelID.DRQA]
# Debugging
modelIds = [
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
# modelIds = [ModelID.TOPIC]

# last ack time, contains datetimes
ack_times = {model: None for model in modelIds}

# TODO: make sure not used in other files before removing
# dumb_qa_model = DumbQuestions_Wrapper('', conf.dumb['dict_file'], ModelID.DUMB_QA)


def start_models():
    """ Warmup models in separate Process
    """
    # Warmup models
    topic = 'user_response'
    job = {'control': 'init', 'topic': topic}
    job_queue.put(job)


def stop_models():
    """ Send command to close the processes first
    """
    topic = 'user_response'
    job = {'control': 'exit', 'topic': topic}
    job_queue.put(job)


def submit_job(job_type='preprocess', to_model=ModelID.ALL,
               context=None, text='', chat_id='', chat_unique_id='',
               article='', all_context=None):
    """ Submit Jobs to job queue, which will be consumed by the responder
    :job_type = preprocess / get_response / exit / discard
    :to_model = all / specific model name
    """
    topic = 'user_response'
    if to_model != ModelID.ALL:
        topic = to_model
    # check if article is spacy instance
    if article and not isinstance(article, basestring):
        article = article.text
    if not context:
        context = []
    job = {'type': job_type, 'topic': topic, 'context': context,
           'text': text, 'chat_id': chat_id, 'chat_unique_id': chat_unique_id,
           'article_text': article, 'all_context': all_context}
    if job_type == 'preprocess' or job_type == 'exit' or job_type == 'discard':
        job['control'] = job_type

    job_queue.put(job)


def act():
    """ On getting a response from a model, add it to the
    chat_id model_responses list.
    Check if chat_id is present in model_responses, if not discard
    """
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(BUS_PIPE)
    logging.info("Child pull channel active")
    while True:
        packet = socket.recv_json()
        if 'control' in packet:
            # received ack response
            ack_times[packet['model_name']] = datetime.now()
        elif packet['chat_unique_id'] in model_responses:
            logging.info("Receiving model response")
            logging.info(packet)
            # calculate the features here
            chat_id = packet['chat_id']
            context = packet['context']
            logging.info("Calculating features for model {}".format(
                packet['model_name']))
            # Now store the packet in dict
            model_responses[packet['chat_unique_id']
                            ][packet['model_name']] = packet
        else:
            logging.info('Discarding message from model {} for chat id {}'.format(
                packet['model_name'], packet['chat_id']))


def responder():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(COMMAND_PIPE)
    logging.info("Child publish channel active")
    while True:
        job = job_queue.get()
        topic = job['topic']
        payload = mogrify(topic, job)
        socket.send(payload)


def consumer():
    """ ZMQ Consumer. Collect jobs from parent and run `get_response`
    """
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(PARENT_PULL_PIPE)
    logging.info("Parent pull channel active")
    is_running = True
    while is_running:
        msg = socket.recv_json()
        if 'control' in msg and msg['control'] != 'test':
            if msg['control'] == 'exit':
                logging.info("Received exit command. Closing all models")
                stop_models()
                logging.info("Exiting")
                is_running = False
                sys.exit(0)
            if msg['control'] == 'clean':
                clean(msg['chat_id'])
        else:
            # Spawn the response generation in new thread
            control = 'none'
            if 'control' in msg:
                control = msg['control']
            gthread = Thread(target=get_response, args=[msg['chat_id'], msg['text'],
                         msg['context'], msg['allowed_model'], control])
            gthread.daemon = True
            gthread.start()


def producer():
    """ ZMQ Response producer. Push response to bot callee.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(PARENT_PIPE)
    logging.info("Parent push channel active")
    while True:
        msg = response_queue.get()
        socket.send_json(msg)
        response_queue.task_done()


def clean(chat_id):
    article_text.pop(chat_id, None)
    candidate_model.pop(chat_id, None)
    article_nouns.pop(chat_id, None)
    boring_count.pop(chat_id, None)
    # policy_mode = Policy.NONE  # TODO: make sure never used before removing

# check the last ack time to determine if the model is dead


def dead_models():
    dm = []
    now = datetime.now()
    for model in ack_times:
        if ack_times[model]:
            diff = now - ack_times[model]
            diff_seconds = diff.total_seconds()
            if diff_seconds > PING_TIME:
                dm.append(model)
    return dm

# check if all models are up


def isEveryoneAwake():
    for model in ack_times:
        if not ack_times[model]:
            return False
    return True


def strip_emojis(str):
    tokens = set(list(str))
    emojis = list(tokens.intersection(set(emoji.UNICODE_EMOJI.keys())))
    if len(emojis) > 0:
        text = ''.join(c for c in str if c not in emojis)
        emojis = ''.join(emojis)
        return text, emojis
    return str, None

# given a set of model_responses, rank the best one based on
# the following policy:
# return ModelID of the model to select, else None
# also return a list of models to NOT consider, else None
# which indicates to take the pre-calculated policy


def ranker(chat_unique_id):
    # array containing tuple of (model_name, rank_score) for 1
    consider_models = []
    dont_consider_models = []  # for 0
    all_models = []  # for debugging purpose
    always_consider = [ModelID.HRED_REDDIT, ModelID.HRED_TWITTER,
                       ModelID.DUAL_ENCODER, ModelID.ALICEBOT]
    logging.info("Ranking among models")
    for model, response in model_responses[chat_unique_id].iteritems():
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

# check if any of the current generated responses fall within k previous history
# If so, remove that response altogether


def no_duplicate(chat_id, chat_unique_id, k=5):
    del_models = []
    for model, response in model_responses[chat_unique_id].iteritems():
        if chat_id in chat_history and response['text'] in set(chat_history[chat_id][-k:]):
            del_models.append(model)
    for dm in del_models:
        del model_responses[chat_unique_id][dm]


def get_response(chat_id, text, context, allowed_model=None, control=None):
    # create a chat_id + unique ID candidate responses field
    # chat_unique_id is needed to uniquely determine the return
    # for each call
    chat_unique_id = str(chat_id) + '_' + str(uuid.uuid4())
    model_responses[chat_unique_id] = {}
    is_start = False
    logging.info("get_response context")
    logging.info(context)

    # if text contains /start, don't add it to the context
    if '/start' in text:
        is_start = True
        # remove start token
        text = re.sub(r'\/start', '', text)
        # remove urls
        text = re.sub(r'https?:\/\/.*[\r\n]*',
                      '', text, flags=re.MULTILINE)
        # save the article for later use
        article_text[chat_id] = text
        article_nlp = nlp(unicode(text))
        # save all nouns from the article
        article_nouns[chat_id] = [
            p.lemma_ for p in article_nlp if p.pos_ in ['NOUN', 'PROPN']
        ]

        # initialize bored count to 0 for this new chat
        boring_count[chat_id] = 0

        # initialize chat history
        chat_history[chat_id] = []

        # initialize model usage history
        used_models[chat_id] = []

        # fire global preprocess call
        submit_job(job_type='preprocess',
                   article=article_text[chat_id],
                   chat_id=chat_id,
                   chat_unique_id=chat_unique_id)
        # fire candidate question and NQG
        submit_job(job_type='get_response',
                   to_model=ModelID.CAND_QA,
                   chat_id=chat_id,
                   chat_unique_id=chat_unique_id,
                   context=context,
                   text='',
                   article=article_text[chat_id],
                   all_context=chat_history[chat_id])
        submit_job(job_type='get_response',
                   to_model=ModelID.NQG,
                   chat_id=chat_id,
                   chat_unique_id=chat_unique_id,
                   context=context,
                   text='',
                   article=article_text[chat_id],
                   all_context=chat_history[chat_id])

    else:
        # fire global query
        if not allowed_model or allowed_model == ModelID.ALL:
            # for test mode only, article_text will not have chat_id
            if control == 'test':
                article = 'preset article'
                all_context = []
            else:
                article = article_text[chat_id]
                all_context = chat_history[chat_id]
            submit_job(job_type='get_response',
                       chat_id=chat_id,
                       chat_unique_id=chat_unique_id,
                       context=context,
                       text=text,
                       article=article,
                       all_context=all_context)
        else:
            submit_job(job_type='get_response',
                       to_model=allowed_model,
                       chat_id=chat_id,
                       chat_unique_id=chat_unique_id,
                       context=context,
                       text=text,
                       article=article_text[chat_id],
                       all_context=chat_history[chat_id])
    # wait for responses to come in
    # if we have answer ready before the wait period, exit and return the answer
    done_processing = False
    wait_for = WAIT_TIME
    # response should be a dict of (text, context, model_name, policy_mode)
    response = {}
    # add feature list as another key of response
    done_features = set()
    while not done_processing and wait_for > 0:
        if is_start:
            if (ModelID.CAND_QA not in
                    model_responses[chat_unique_id]
                ) and (ModelID.NQG not in
                       model_responses[chat_unique_id]):
                continue
            else:
                # if found msg early, break
                done_processing = True
                break
        else:
            # Only for debugging
            if allowed_model and allowed_model != ModelID.ALL:
                if allowed_model in model_responses[chat_unique_id]:
                    done_processing = True
                    break
            # Wait for atleast FACT_GEN to arrive
            elif ModelID.FACT_GEN not in set(model_responses[chat_unique_id].keys()):
                done_processing = False
                wait_for += 1
            # Wait for all the models to arrive - REDUNDANT
            elif len(set(model_responses[chat_unique_id].keys())
                     .intersection(set(modelIds))) == len(modelIds):
                done_processing = True
                break
        # tick
        wait_for -= 1
        time.sleep(1)

    logging.info("Received responses from {}".format(
        model_responses[chat_unique_id].keys()))
    if chat_id not in boring_count:
        boring_count[chat_id] = 0
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    if chat_id not in used_models:
        used_models[chat_id] = []
    # instruct models to not further process
    submit_job(job_type='discard', chat_id=chat_id, chat_unique_id=chat_unique_id)
    
    # got the responses, now choose which one to send.
    if is_start:
        # TODO: replace this with a proper choice / always NQG?
        choices = list(set([ModelID.CAND_QA, ModelID.NQG])
                       .intersection(set(model_responses[chat_unique_id].keys())))
        if len(choices) > 0:
            selection = random.choice(choices)
            response = model_responses[chat_unique_id][selection]
            response['policyID'] = Policy.START
    else:
        # check if allowed_model is set, then only reply from the allowed
        # model. This is done for debugging.
        # TODO: Probably remove this before final submission?
        if allowed_model and allowed_model != ModelID.ALL:
            response = model_responses[chat_unique_id][allowed_model]
            response['policyID'] = Policy.FIXED
        else:
            # if text contains emoji's, strip them
            text, emojis = strip_emojis(text)
            # check if the text contains wh words
            ntext = nlp(unicode(text))
            nt_words = [p.lemma_ for p in ntext]
            has_wh_word = False
            for word in nt_words:
                if word in set(conf.wh_words):
                    has_wh_word = True
                    break
            if emojis and len(text.strip()) < 1:
                # if text had only emoji, give back the emoji itself
                response = {'response': emojis, 'context': context,
                            'model_name': 'emoji', 'policy': Policy.NONE}

            # if query falls under dumb questions, respond appropriately
            elif ModelID.DUMB_QA in model_responses[chat_unique_id]:
                logging.info("Matched dumb preset patterns")
                response = model_responses[
                    chat_unique_id][ModelID.DUMB_QA]
                response['policyID'] = Policy.FIXED
            # if query falls under topic request, respond with the article topic
            elif ModelID.TOPIC in model_responses[chat_unique_id]:
                logging.info("Matched topic preset patterns")
                response = model_responses[
                    chat_unique_id][ModelID.TOPIC]
                response['policyID'] = Policy.FIXED
            # if query is a question, try to reply with DrQA
            elif has_wh_word or ("which" in set(nt_words)
                                 and "?" in set(nt_words)):
                # get list of common nouns between article and question
                if chat_id in article_nouns:
                    common = list(set(article_nouns[chat_id]).intersection(
                        set(nt_words)))
                else:
                    common = []
                logging.info(
                    'Common nouns between question and article: {}'.format(common))
                # if there is a common noun between question and article
                # select DrQA
                if len(common) > 0 and ModelID.DRQA in model_responses[chat_unique_id]:
                    response = model_responses[
                        chat_unique_id][ModelID.DRQA]
                    response['policyID'] = Policy.FIXED

    if not response:
        # remove duplicates responses from k nearest chats
        no_duplicate(chat_id, chat_unique_id)
        # Ranker based selection
        best_model, dont_consider = ranker(chat_unique_id)
        if dont_consider and len(dont_consider) > 0:
            for model, score in dont_consider:
                # remove the models from futher consideration
                del model_responses[chat_unique_id][model]

        # Reduce confidence of CAND_QA
        if ModelID.CAND_QA in model_responses[chat_unique_id]:
            cres = model_responses[chat_unique_id][ModelID.CAND_QA]
            cres_conf = float(cres['conf'])
            cres['conf'] = str(cres_conf / 2) # half the confidence
            model_responses[chat_unique_id][ModelID.CAND_QA] = cres

        # Bored model selection (TODO: nlp() might be taking time)
        nt_sent = nlp(unicode(text))
        nt_words = [p.lemma_ for p in nt_sent]
        # check if user said only generic words:
        generic_turn = True
        for word in nt_words:
            if word not in generic_words_list:
                generic_turn = False
                break
        # if text contains 2 words or less, add 1 to the bored count
        # also consider the case when the user says only generic things
        if len(text.strip().split()) <= 2 or generic_turn:
            boring_count[chat_id] += 1
        # list of available models to use if bored
        bored_models = [ModelID.NQG, ModelID.FACT_GEN,
                        ModelID.CAND_QA, ModelID.HUMAN_IMITATOR]
        boring_avl = list(
            set(model_responses[chat_unique_id]).intersection(set(bored_models)))
        # if user is bored, change the topic by asking a question
        # (only if that question is not asked before)
        if boring_count[chat_id] >= BORED_COUNT and len(boring_avl) > 0:
            # assign model selection probability based on estimator confidence
            confs = [float(model_responses[chat_unique_id][model]['conf'])
                     for model in boring_avl]
            norm_confs = confs / np.sum(confs)
            selection = np.random.choice(boring_avl, 1, p=norm_confs)[0]
            response = model_responses[chat_unique_id][selection]
            response['policyID'] = Policy.BORED
            boring_count[chat_id] = 0  # reset bored count to 0

        # If not bored, then select from best model
        elif best_model:
            response = model_responses[chat_unique_id][best_model]
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
                set(model_responses[chat_unique_id]).intersection(models))
            if len(available_models) > 0:
                # assign model selection probability based on estimator confidence
                confs = [float(model_responses[chat_unique_id][model]['conf'])
                         for model in available_models]
                norm_confs = confs / np.sum(confs)
                chosen_model = np.random.choice(
                    available_models, 1, p=norm_confs)
                response = model_responses[chat_unique_id][chosen_model[0]]
                response['policyID'] = Policy.OPTIMAL

    # if still no response, then just send a random fact
    if not response or 'text' not in response:
        logging.warn("Failure to obtain a response, using fact gen")
        response = model_responses[chat_unique_id][ModelID.FACT_GEN]
        response['policyID'] = Policy.FIXED

    # Now we have a response, so send it back to bot host
    # add user and response pair in chat_history
    chat_history[response['chat_id']].append(response['context'][-1])
    chat_history[response['chat_id']].append(response['text'])
    used_models[chat_id].append(response['model_name'])
    # Again use ZMQ, because lulz
    response['control'] = control
    logging.info("Done selecting best model")
    response_queue.put(response)
    # clean the unique chat ID
    del model_responses[chat_unique_id]


if __name__ == '__main__':
    """Run the main calling function:
            1. Initialize all the models
            2. Bot parent push channel `producer`
            3. Bot parent pull channel `consumer`
            4. Child models publish channel `responder`
            5. Child models pull channel `act`
    """
    # 1. Initializing the models
    process_manager = {}
    for model in modelIds:
        # wx = ModelClient(model)
        # mps.append(wx)
        process_manager[model] = Process(target=ModelClient, args=(model,))
        process_manager[model].start()
    # for mp in mps:
    #    mp.start()
    # 2. Parent -> Bot publish channel
    child_publish_thread = Thread(target=responder)
    child_publish_thread.daemon = True
    child_publish_thread.start()
    # 3. Bot -> Parent push channel
    child_pull_thread = Thread(target=act)
    child_pull_thread.daemon = True
    child_pull_thread.start()
    # 4. Parent -> Callee push channel
    parent_push_thread = Thread(target=producer)
    parent_push_thread.daemon = True
    parent_push_thread.start()
    # 5. Callee -> Parent pull channel
    parent_pull_thread = Thread(target=consumer)
    parent_pull_thread.daemon = True
    parent_pull_thread.start()

    # Model Init
    start_models()

    all_awake = False

    try:
        while True:
            dm = dead_models()
            if not all_awake and isEveryoneAwake():
                logging.info("====================================")
                logging.info("======RLLCHatBot Active=============")
                logging.info("All modules of the bot has been loaded.")
                logging.info("Thanks for your patience")
                logging.info("-------------------------------------")
                logging.info("Made with <3 in Montreal")
                logging.info("Reasoning & Learning Lab, McGill University")
                logging.info("Fall 2017")
                logging.info("=====================================")
                all_awake = True

            time.sleep(120)
            if len(dm) > 0:
                for dead_m in dm:
                    logging.info("Reviving model {}".format(dead_m))
                    process_manager[dead_m] = Process(
                        target=ModelClient, args=(dead_m,))
                    process_manager[dead_m].start()
        # doesn't matter to wait for join now?
        # mp_pool.close()
        # mp_pool.join()
        # for mp in mps:
        #    mp.join()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Sending shutdown signal to all models")
        stop_models()
        for model in modelIds:
            process_manager[model].terminate()
        logging.info("Shutting down master")
