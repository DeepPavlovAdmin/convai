import random
import subprocess
import logging
from nltk import word_tokenize
from skills.utils import combinate_and_return_answer, get_stopwords_count


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationSkill:
    """Responses with summary of a given wiki-text"""
    def __init__(self, summarization_url, text):
        self._summarization_url = summarization_url
        self._text = text

    def predict(self, with_heuristic=True):
        return self._get_summaries(with_heuristic)

    def _get_summaries(self, with_heuristic=True):
        text = self._text
        if not text:
            return None
        logger.info("Send to opennmt summary: {}".format(text))
        # TODO: Remove dependencies on from_* folders;
        cmd = "echo \"{}\" | python from_opennmt_summary/get_reply.py {}".format(text, self._summarization_url)
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        res = str(output, "utf-8").strip()
        logger.info("Got from opennmt summary: {}".format(res))
        # now lets select one best response
        candidates = []
        for line in res.split('\n'):
            _, resp, score = line.split('\t')
            words_cnt = len(word_tokenize(resp))
            logger.info("Summarization skill info: {} {} {}".format(resp, words_cnt, get_stopwords_count(resp)))
            if words_cnt >= 2 and get_stopwords_count(resp) / words_cnt < 0.5 and '<unk>' not in resp:
                candidates.append(resp)
        if len(candidates) > 0:
            summary = random.choice(candidates)
            msg1 = ['I think this', 'I suppose that this', 'Maybe this']
            msg2 = ['article', 'text', 'paragraph']
            msg3 = ['can be described as:', 'can be summarized as:', 'main idea is:', 'in a nutshell is:']
            msg4 = [summary]
            msg5 = ['.', '...', '?', '..?']
            msg = [msg1, msg2, msg3, msg4, msg5]
            return combinate_and_return_answer(msg)
        return None
