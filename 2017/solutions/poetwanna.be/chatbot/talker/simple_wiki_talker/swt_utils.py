from random import randint, choice
from string import punctuation
from utils import U


greetings = {'hello', 'hi', 'hi there', 'good morning', 'hello!', 'hi!',
             'hi there!', 'good morning!', 'goodbye', 'bye'}

popular = {'no', 'yes', 'sure', 'bye', 'nope', 'yeah', 'yep', 'he', 'she',
           'ok', 'his', 'him', 'me', 'i', 'you', 'her', 'they', 'we', 'them',
           'it'}


class ZeroDict(dict):

    def __getitem__(self, key):
        if key not in self:
            self[key] = 0
        return super(ZeroDict, self).__getitem__(key)


def lower_string(s):
    return U(s).lower().encode('utf8')


def title_key(s):
    return s.replace(' ', '_')


def key_title(s):
    return s.replace('_', ' ')


def word_tokenize(s):
    r = [c for c in s.strip() if c not in punctuation]
    return ''.join(r).split()


def quote(s):
    if ' ' in s:
        return '"' + s + '"'
    return s


def let_us_repeat():
    return 'Let us repeat: '


followup_intros = [
    'I guess you probably will be interested in %s or %s.',
    'Maybe %s or %s will be interesting to you.',
    'I can tell you about %s. Or about %s.',
    '%s or %s. Your choice!',
    'There are many interesting things related to it. Say %s, or %s.',
    "I've heard about %s and %s. Have you?",
    "Maybe we can talk about %s or %s?",
    "%s is an interesting topic, and %s is even better.",
    "Why don't we talk about %s? Why don't we talk about %s?",
    "Someone told me about %s and %s. Maybe now I can discuss it with you?",
    "Choose your favourite topic: %s or %s.",
    "Let's talk about %s or %s. Why not?",
    "These two things also seem important: %s and %s."
]


def follow_up_text(s1, s2):
    s1 = quote(key_title(s1))
    s2 = quote(key_title(s2))
    text = choice(followup_intros)
    return text % (s1, s2)


def topic_intro(s):
    return quote(s) + ' ' + choice(
        ['is really an interesting topic.', 'is worth to talk about.'])


def strange_topic(s):
    return 'Well, I will think about ' + s + 'in the future...'


def strange_title(s):
    return 'It is really weird thing: ' + s
