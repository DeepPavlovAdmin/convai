import random
import logging
import requests


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicDetectionSkill:
    """Responses with topic of a given text"""
    def __init__(self, topic_url, text):
        if text:
            r = requests.post(topic_url + '/respond', json={'text': text})
            topics_info = r.json()['result']
            logger.info("Topics result: {}".format(topics_info))
            self._topic_responses = topics_info[0]['responses']
        else:
            self._topic_responses = [None]

    def predict(self, argument=None):
        return random.sample(self._topic_responses, k=1)[0]
