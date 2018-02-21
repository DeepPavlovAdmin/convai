# Script to select model conversation
# Initialize all the models here and then reply with the best answer
# some logic-foo need to be done here.
import config
conf = config.get_config()
from models.wrapper import HRED_Wrapper, Dual_Encoder_Wrapper, HREDQA_Wrapper, CandidateQuestions_Wrapper, DumbQuestions_Wrapper, DRQA_Wrapper
import random
import copy
import spacy
import re
import time
import emoji

nlp = spacy.load('en')


class Policy:
    NONE = -1        # between each chat
    OPTIMAL = 1      # current set of rules
    # exploratory policies:
    HREDx2 = 2       # hred-reddit:0.5 & hred-twitter:0.5
    HREDx3 = 3       # hred-reddit:0.33 & hred-twitter:0.33 & hred-qa:0.33
    HREDx2_DE = 4    # hred-reddit:0.25 & hred-twitter:0.25 & DualEncoder:0.5
    HREDx2_DRQA = 5  # hred-reddit:0.25 & hred-twitter:0.25 & DrQA:0.5
    DE_DRQA = 6      # DualEncoder:0.5 & DrQA:0.5


ALL_POLICIES = [Policy.OPTIMAL, Policy.HREDx2, Policy.HREDx3,
                Policy.HREDx2_DE, Policy.HREDx2_DRQA, Policy.DE_DRQA]

BORED_COUNT = 2


class ModelSelection(object):

    def __init__(self):
        self.article_text = {}     # map from chat_id to article text
        self.candidate_model = {}  # map from chat_id to a simple model for the article
        self.article_nouns = {}    # map from chat_id to a list of nouns in the article
        self.boring_count = {}     # number of times the user responded with short answer
        self.policy_mode = Policy.NONE

    def initialize_models(self):
        self.hred_model_twitter = HRED_Wrapper(
            conf.hred['twitter_model_prefix'], conf.hred['twitter_dict_file'], 'hred-twitter')
        self.hred_model_reddit = HRED_Wrapper(
            conf.hred['reddit_model_prefix'], conf.hred['reddit_dict_file'], 'hred-reddit')
        self.de_model_reddit = Dual_Encoder_Wrapper(
            conf.de['reddit_model_prefix'], conf.de['reddit_data_file'], conf.de['reddit_dict_file'], 'de-reddit')
        self.qa_hred = HREDQA_Wrapper(
            conf.followup['model_prefix'], conf.followup['dict_file'], 'followup_qa')
        self.dumb_qa = DumbQuestions_Wrapper(
            '', conf.dumb['dict_file'], 'dumb_qa')
        self.drqa = DRQA_Wrapper('', '', 'drqa')

        # warmup the models before serving
        r, _ = self.hred_model_twitter.get_response(1, 'test statement', [])
        r, _ = self.hred_model_reddit.get_response(1, 'test statement', [])
        r, _ = self.qa_hred.get_response(1, 'test statement', [])
        r, _ = self.de_model_reddit.get_response(1, 'test statement', [])
        r, _ = self.drqa.get_response(
            1, 'Where is Daniel?', [], nlp(unicode('Daniel went to the kitchen')))

    def clean(self, chat_id):
        del self.article_text[chat_id]
        del self.candidate_model[chat_id]
        del self.article_nouns[chat_id]
        del self.boring_count[chat_id]
        self.policy_mode = Policy.NONE

    def strip_emojis(self, str):
        tokens = set(list(str))
        emojis = list(tokens.intersection(set(emoji.UNICODE_EMOJI.keys())))
        if len(emojis) > 0:
            text = ''.join(c for c in str if c not in emojis)
            emojis = ''.join(emojis)
            return text, emojis
        return str, None

    def get_response(self, chat_id, text, context):
        # if text contains /start, don't add it to the context
        if '/start' in text:
            # Make sure we didn't sample a policy before
            assert self.policy_mode == Policy.NONE
            self.policy_mode = random.choice(ALL_POLICIES)  # sample a random policy

            # remove start token
            text = re.sub(r'\/start', '', text)
            # remove urls
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            # save the article for later use
            self.article_text[chat_id] = nlp(unicode(text))
            # save all nouns from the article
            self.article_nouns[chat_id] = [
                p.text for p in self.article_text[chat_id] if p.pos_ == 'NOUN'
            ]
            print self.article_nouns[chat_id]

            # Initialize candidate model, responsible for generating a question on the article
            try:
                self.candidate_model[chat_id] = CandidateQuestions_Wrapper('', self.article_text[chat_id],
                                                                       conf.candidate['dict_file'], 'candidate_question')
            except Exception as e:
                logger.error('Exception in candidate model init')
                logger.error(str(e))

            # initialize bored count to 0 for this new chat
            self.boring_count[chat_id] = 0

            # add a small delay
            time.sleep(2)

            # Try to generate a question on this article
            resp = ''
            try:
                if self.candidate_model[chat_id]: # make sure we initialized the model before
                    resp, context = self.candidate_model[chat_id].get_response(
                        chat_id, '', context)
            except Exception as e:
                logger.error('Error in generating candidate response')
                logger.error(str(e))

            if resp == '':
                resp = random.choice(["That's a short article, don't you think? Not sure what's it about.",
                                      "Apparently I am too dumb for this article. What's it about?"])
                # append random response to context here since candidate_model wasn't able to do it
                context.append('<first_speaker>' + resp + '</s>')

            # return generated response, new context (contains `resp`), model_name, policy
            return (resp, context, 'starter'), self.policy_mode

        # if text contains emoji's, strip them
        text, emojis = self.strip_emojis(text)
        if emojis and len(text.strip()) < 1:
            # if text had only emoji, give back the emoji itself
            # NOTE: shouldn't we append the `resp` (in this case emoji) to the context like everywhere else?
            return (emojis, context, 'emoji'), self.policy_mode

        # if query falls under dumb questions, respond appropriately
        if self.dumb_qa.isMatch(text):
            # generate response and update context
            resp, context = self.dumb_qa.get_response(chat_id, text, context)
            # return generated response, new context (contains `resp`), model_name, policy
            return (resp, context, 'dumb qa'), self.policy_mode

        ###
        # chat selection logic
        ###
        assert self.policy_mode != Policy.NONE

        if self.policy_mode == Policy.OPTIMAL:
            return self.optimal_policy(chat_id, text, context), Policy.OPTIMAL
        elif self.policy_mode == Policy.HREDx2:
            return self.hredx2_policy(chat_id, text, context), Policy.HREDx2
        elif self.policy_mode == Policy.HREDx3:
            return self.hredx3_policy(chat_id, text, context), Policy.HREDx3
        elif self.policy_mode == Policy.HREDx2_DE:
            return self.hredx2_de_policy(chat_id, text, context), Policy.HREDx2_DE
        elif self.policy_mode == Policy.HREDx2_DRQA:
            return self.hredx2_drqa_policy(chat_id, text, context), Policy.HREDx2_DRQA
        elif self.policy_mode == Policy.DE_DRQA:
            return self.de_drqa_policy(chat_id, text, context), Policy.DE_DRQA
        else:
            print "ERROR: unknown policy mode:", self.policy_mode
            return (None, None, None), None

    def optimal_policy(self, chat_id, text, context):
        # if text contains question,
        if '?' in text:
            # get list of common nouns between article and question
            common = list(set(self.article_nouns[chat_id]).intersection(
                set(text.split(' '))))
            print 'common nouns between question and article:', common
            # if there is a common noun between question and article, run DrQA
            if len(common) > 0:
                resp, context = self.drqa.get_response(
                    chat_id, text, context, article=self.article_text[chat_id].text)
                return resp, context, 'drqa'

        # if text contains 2 words or less, add 1 to the bored count
        if len(text.strip().split()) <= 2:
            self.boring_count[chat_id] += 1
        # if user is bored, change the topic by asking a question (only if that question is not asked before)
        if self.boring_count[chat_id] >= BORED_COUNT:
            resp_c, context_c = self.candidate_model[chat_id].get_response(
                chat_id, '', copy.deepcopy(context))
            if resp_c != '':
                self.boring_count[chat_id] = 0  # reset bored count to 0
                return resp_c, context_c, 'bored'

        # randomly decide a model to query to get a response:
        models = ['hred-twitter', 'hred-reddit', 'de']
        if '?' in text:
            # if the user asked a question, also consider DrQA
            models.append('drqa')
        else:
            # if the user didn't ask a question, also consider hred-qa
            models.append('qa')

        chosen_model = random.choice(models)
        origin_context = copy.deepcopy(context)
        if chosen_model == 'hred-twitter':
            resp, cont = self.hred_model_twitter.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'hred-reddit':
            resp, cont = self.hred_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'de':
            resp, cont = self.de_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'qa':
            resp, cont = self.qa_hred.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'drqa':
            resp, cont = self.drqa.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        else:
            print "ERROR: unknown chosen model:", chosen_model
            return None, None, None

        return resp, cont, chosen_model

    def hredx2_policy(self, chat_id, text, context):
        # randomly decide a model to query to get a response:
        models = ['hred-twitter', 'hred-reddit']
        chosen_model = random.choice(models)
        origin_context = copy.deepcopy(context)

        if chosen_model == 'hred-twitter':
            resp, cont = self.hred_model_twitter.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'hred-reddit':
            resp, cont = self.hred_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        else:
            print "ERROR: unknown chosen model:", chosen_model
            return None, None, None

        return resp, cont, chosen_model

    def hredx3_policy(self, chat_id, text, context):
        # randomly decide a model to query to get a response:
        models = ['hred-twitter', 'hred-reddit', 'qa']
        chosen_model = random.choice(models)
        origin_context = copy.deepcopy(context)

        if chosen_model == 'hred-twitter':
            resp, cont = self.hred_model_twitter.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'hred-reddit':
            resp, cont = self.hred_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'qa':
            resp, cont = self.qa_hred.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        else:
            print "ERROR: unknown chosen model:", chosen_model
            return None, None, None

        return resp, cont, chosen_model

    def hredx2_de_policy(self, chat_id, text, context):
        # randomly decide a model to query to get a response:
        # DE has probability .5, hred-reddit .25, hred-twitter .25
        models = ['hred-twitter', 'hred-reddit', 'de', 'de']
        chosen_model = random.choice(models)
        origin_context = copy.deepcopy(context)

        if chosen_model == 'hred-twitter':
            resp, cont = self.hred_model_twitter.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'hred-reddit':
            resp, cont = self.hred_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'de':
            resp, cont = self.de_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text[chat_id])
        else:
            print "ERROR: unknown chosen model:", chosen_model
            return None, None, None

        return resp, cont, chosen_model

    def hredx2_drqa_policy(self, chat_id, text, context):
        # randomly decide a model to query to get a response:
        # DrQA has probability .5, hred-reddit .25, hred-twitter .25
        models = ['hred-twitter', 'hred-reddit', 'drqa', 'drqa']
        chosen_model = random.choice(models)
        origin_context = copy.deepcopy(context)

        if chosen_model == 'hred-twitter':
            resp, cont = self.hred_model_twitter.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'hred-reddit':
            resp, cont = self.hred_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'drqa':
            resp, cont = self.drqa.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        else:
            print "ERROR: unknown chosen model:", chosen_model
            return None, None, None

        return resp, cont, chosen_model

    def de_drqa_policy(self, chat_id, text, context):
        # randomly decide a model to query to get a response:
        models = ['de', 'drqa']
        chosen_model = random.choice(models)
        origin_context = copy.deepcopy(context)

        if chosen_model == 'de':
            resp, cont = self.de_model_reddit.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        elif chosen_model == 'drqa':
            resp, cont = self.drqa.get_response(
                chat_id, text, origin_context, self.article_text.get(chat_id, ''))
        else:
            print "ERROR: unknown chosen model:", chosen_model
            return None, None, None

        return resp, cont, chosen_model
