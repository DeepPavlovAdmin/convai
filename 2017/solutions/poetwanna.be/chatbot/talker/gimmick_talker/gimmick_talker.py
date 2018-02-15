from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

import codecs
import config
import talker.base
import utils
import unicodedata
from os.path import join
from talker.gimmick_talker.langid import LanguageIdentifier, model
import re

langid_languages = """
    af am an ar as az be bg bn br bs ca cs cy da de dz
    el en eo es et eu fa fi fo fr ga gl gu he hi hr ht
    hu hy id is it ja jv ka kk km kn ko ku ky la lb lo
    lt lv mg mk ml mn mr ms mt nb ne nl nn no oc or pa
    pl ps pt qu ro ru rw se si sk sl sq sr sv sw ta te
    th tl tr ug uk ur vi vo wa xh zh zu""".strip().split()


def force_unicode(s):
    if isinstance(s, unicode):
        return s
    else:
        return codecs.decode(s, 'utf8')


def draw_sample(l):
    return l[np.random.randint(len(l))]


def is_punc(c):
    return unicodedata.category(c).startswith('P')


def del_punctuation(text):
    return "".join(x for x in text if not is_punc(x))


class URLGimmick(object):
    first_time_confidence = 11.0
    confidence = 1.0

    def __init__(self):
        # Regular expression dapted from: https://stackoverflow.com/a/31952097
        regex = r'('
        regex += r'(?:(^|\s))'
        # Scheme (HTTP, HTTPS, FTP and SFTP) or www (to ensure user acutally
        # ment some url):
        regex += r'((?:(https?|s?ftp):\/\/)|(?:www\.))'
        regex += r'('
        # Host and domain
        regex += r'(?:(?:[A-Z0-9][A-Z0-9-]{0,61}[A-Z0-9]\.)+)'
        regex += r'([A-Z]{2,6})'  # TLD
        regex += r'|(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP Address
        regex += r')'
        regex += r'(?::(\d{1,5}))?'  # Port
        regex += r'(?:(\/\S+)*)'  # Query path
        regex += r'(?:($|\s))'
        regex += r')'

        self.compiled_regex = re.compile(regex, re.IGNORECASE | re.UNICODE)
        self.responses = [
            u"I was taught to know better than to open links from strangers :("
            " You can always tell me what it is about",
            u"I'm afraid to open links from strangers. Call me paranoid but "
            "I'm not doing it.",
        ]

    def __call__(self, text):
        if self.compiled_regex.search(text) is not None:
            return draw_sample(self.responses)

        return None


class MailGimmick(object):
    first_time_confidence = 11.0
    confidence = 1.0

    def __init__(self):
        regex = r'('
        regex += r'(?:^|\s)'
        regex += r'(?:\w+(?:\.\w+)*)'
        regex += r'(?:@)'
        regex += r'(?:\w+(?:\.\w+)*)'
        regex += r'(?:\.[A-Z]{2,6})'
        regex += r'(?:$|\s)'
        regex += r')'

        self.compiled_regex = re.compile(regex, re.IGNORECASE)
        self.responses = [
            u"I will add this address to my contacts.",
            u"My email address is botty@mcbotface.bt",
            u"I didn't see that address before! It's always "
            "good to make new friends.",
            u"Is that an email address? I gotta check if it works!"
        ]

    def __call__(self, text):
        if self.compiled_regex.search(text) is not None:
            return draw_sample(self.responses)

        return None


class OneLetterGimmick(object):
    first_time_confidence = 11.0
    confidence = 1.0
    letters = "abcdefghijklmnopqrstuvwxyz"

    def __call__(self, text, thresh=5):
        text = ''.join(text.split())
        text_len = len(text)
        if len(set(text)) == 1 and text_len >= thresh:
            response_letter = draw_sample(self.letters.replace(text[0], ''))

            dev = int(text_len/5)
            response_len = np.random.randint(
                max(thresh, text_len-dev), text_len+dev)

            return response_len * response_letter

        return None


class NotAWordGimmick(object):
    first_time_confidence = 11.0
    confidence = 1.0

    def __call__(self, text, thresh=20):
        text = text.strip()
        for word in text.split():
            if len(word) >= thresh:
                return "Are you sure %s is an actual word?" % word

        return None


class EmojiGimmick(object):
    first_time_confidence = 11.0
    confidence = 1.0

    def __init__(self):
        self.emoji2class, self.classes, self.emoji_set = self.load_emoji()

    def __call__(self, text):
        intersection = set(text) & self.emoji_set

        if intersection:
            response = []
            for emoji in intersection:
                response.append(
                    draw_sample(self.classes[self.emoji2class[emoji]]))

            return " ".join(response)

        return None

    def load_emoji(self):
        emoji2class = {}
        classes = {}

        with open(config.emoji_data, 'r') as f:
            for line in f:
                line = line.decode('utf-8')
                emoji, name, subgroup = line.strip().split("\t")
                emoji2class[emoji] = subgroup
                if not subgroup in classes:
                    classes[subgroup] = []
                classes[subgroup].append(emoji)

        emoji_set = set(emoji2class.keys())

        return emoji2class, classes, emoji_set


class PolyglotGimmick(object):
    first_time_confidence = 11.0
    confidence = 1.0

    def __init__(self):
        self.identifier = LanguageIdentifier.from_modelstring(
            model, norm_probs=True)
        self._load_data()

    def __call__(self, text):
        lang = self.detect_lang(text)
        if lang:
            if lang in self.lang2lang_name:
                lang_name = self.lang2lang_name[lang]
            else:
                lang_name = None

            response = [self.foreign_response(lang)]

            if response and lang_name:
                response.append(u"It's so much I know in %s :( " % lang_name)
            response.append(u"Let's speak in English from now on, shall we?")
            return " ".join(response)

        return None

    def foreign_response(self, lang):
        print(lang)
        response = []
        if lang in self.lang2greetings:
            response.append(
                self.num2sent[draw_sample(self.lang2greetings[lang])])
        if lang in self.lang2do_u_english:
            response.append(
                self.num2sent[draw_sample(self.lang2do_u_english[lang])])
        return " ".join(response)

    def detect_lang(self, text):
        text_no_punc = del_punctuation(text.lower())

        if len(text_no_punc) < 5:
            return None

        text_set = set(text_no_punc.split())
        langs = [lang for lang in self.lang2greetings if len(lang) == 3]
        candidates = [
            lang for lang in langs
            if any(map(
                lambda num: bool(text_set & self.num2sent_clean[num]),
                self.lang2greetings[lang+"deep"]))
        ]

        if 'eng' in candidates or not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        langs = [self.lang2id[lang] for lang in candidates
                 if lang in self.lang2id and
                 self.lang2id[lang] in langid_languages]
        self.identifier.set_languages(langs)
        language, conf = self.identifier.classify(text)
        language = self.id2lang[language] if language in self.id2lang else None

        if (language != 'eng') and (conf > 1.0/len(langs)):
            return language

        return None

    def _load_data(self):
        path = config.polyglot_data_path

        num2sent = {}
        num2sent_clean = {}
        num2lang = {}
        lang2greetings = {}
        lang2do_u_english = {}
        lang2lang_name = {}
        id2lang = {}

        with open(join(path, "sentences_light.txt"), 'r') as f:
            for line in f:
                num, lang, sent = line.strip().decode("utf-8").split("\t")
                num = int(num)
                num2sent[num] = sent
                num2sent_clean[num] = set(
                    del_punctuation(sent.lower()).split())
                num2lang[num] = lang

        with open(join(path, "greetings.txt"), 'r') as f:
            for line in f:
                line_split = line.strip().decode("utf-8").split("\t")
                lang = line_split[0]
                lang2greetings[lang] = list(map(int, line_split[1:]))

        with open(join(path, "do_u_english.txt"), 'r') as f:
            for line in f:
                line_split = line.strip().decode("utf-8").split("\t")
                lang = line_split[0]
                lang2do_u_english[lang] = list(map(int, line_split[1:]))

        with open(join(path, "iso-639-3.tab"), 'r') as f:
            iso_lines = f.readlines()[1:]
            for line in iso_lines:
                line_split = line.strip().decode('utf-8').split("\t")
                iso639_3 = line_split[0]
                iso639_1 = line_split[3]
                lang_name = line_split[-1]

                lang2lang_name[iso639_3] = lang_name
                if iso639_1:
                    id2lang[iso639_1] = iso639_3

        self.num2sent = num2sent
        self.num2sent_clean = num2sent_clean
        self.num2lang = num2lang
        self.lang2greetings = lang2greetings
        self.lang2do_u_english = lang2do_u_english
        self.lang2lang_name = lang2lang_name
        self.id2lang = id2lang
        self.lang2id = {id2lang[lid]: lid for lid in id2lang}


class GimmickTalker(talker.base.ResponderRole):
    name = "gimmick_talker"
    default_confidence = 0.0

    def __init__(self, **kwargs):
        super(GimmickTalker, self).__init__(**kwargs)
        self.gimmicks = [
            URLGimmick(),
            MailGimmick(),
            OneLetterGimmick(),
            NotAWordGimmick(),
            EmojiGimmick(),
            PolyglotGimmick(),
        ]

        self.gimmicks_no = len(self.gimmicks)
        self.default_responses = [
            "Well...",
            "Hmm..",
            "mhm",
            "if you say so...",
        ]

    def new_state(self):
        probs = [1.0 for _ in range(self.gimmicks_no)]
        return probs

    def _respond_to(self, state,
                    last_user_utt_dict, last_bot_utt, user_utt_dict):
        del last_user_utt_dict  # unused

        user_utt = user_utt_dict['raw_utt'] or ''
        user_utt = force_unicode(user_utt)

        response = draw_sample(self.default_responses)
        confidence = self.default_confidence
        prob_thresh = np.random.random()

        for i, gimmick in enumerate(self.gimmicks):
            res = gimmick(user_utt)
            prob = state[i]
            if res is not None and prob > prob_thresh:
                response = res
                if prob == 1.0:
                    confidence = gimmick.first_time_confidence
                else:
                    confidence = gimmick.confidence
                state[i] /= 2.0
                break

        return (state, response, confidence)


if __name__ == "__main__":
    prev = u""
    talker = GimmickTalker()

    print("Enter article")

    state = talker.new_state()
    state = talker.set_article(state, dict(text=raw_input().decode('utf-8')))
    print("System ready, start talking")
    while True:
        x = raw_input("Me: ").decode('utf-8')
        print("Me: %s" % repr(x))
        state, txt, val = talker._respond_to(state, {}, prev, dict(raw_utt=x))
        prev = txt
        print('Bot [%.2f]: %s' % (val, repr(txt)))
