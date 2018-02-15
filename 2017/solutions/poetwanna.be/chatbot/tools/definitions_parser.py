from string import punctuation


DEF_STR = """
  sure, tell me about
  tell me about
  tell me more about
  could you please tell me about
  what's
  what 's
  what is
  what is the
  what are
  what are the
  what is a
  what is an
  what about
  what was
  what were
  who's
  who 's
  who is
  who are
  who was
  who were
  what is the definition of
  what is the meaning of
  give me a description
  give me a definition
  give me description
  give me definition
"""

QUESTION_WORD = 'what when did do done have had been you he she it are is '
'were which who why whom whose'.split()

# "i" is omitted -- it is to short, "us" is to similar to United States
PRONOUNS = set('me we you she her he him his it'.split())
PRONOUNS |= set('that which who whom whose whichever whoever whomever this '
                'that those these'.split())
# one is omitted
PRONOUNS |= set(
    'anybody anyone anything each either everybody everyone '
    'everything neither nobody nothing somebody someone something'.split())
PRONOUNS |= set('both few many several'.split())
PRONOUNS |= set('all any most none some'.split())
PRONOUNS |= set('myself yourself ourselves  yourselves himself herself '
                'itself themselves'.split())
# mine is omitted
PRONOUNS |= set('my our your his her its their hers theirs'.split())

FORBIDDEN_PRONOUNS = set(
    "i me you he him she her it we they them "  # us = United States
    "this those these "
    "myself yourself himself herself itself ourselves themselves yourselves "
    "my your her his its their our "
    "mine yours hers ours theirs "
    "i'm you're he's she's it's we're they're".split())

SINGLE_WORD_PREPOSITIONS = set("""
    aboard about above across
    after against along among
    around as at before
    behind below beneath beside
    between beyond but by
    despite down during except
    failing following for from
    in inside into like
    minus near next of
    off on onto opposite
    out outside over past
    plus regarding since than
    through throughout till to
    toward towards under underneath
    unlike until up upon
    via with within without
""".split())


def_phrases = [x.strip() for x in DEF_STR.split('\n')]
def_phrases = [p for p in def_phrases if len(p) > 0]

new_def_phrases = ["ok, " + dp for dp in def_phrases]
new_def_phrases += ["ok then, " + dp for dp in def_phrases]
new_def_phrases += ["ok, " + dp for dp in def_phrases]
new_def_phrases += ["well then, " + dp for dp in def_phrases]

def_phrases.extend(new_def_phrases)
def_phrases.sort(key=len, reverse=True)


def clear(s):
    return ' '.join(s.split())


def delete_def_prefix(s):
    for d in def_phrases:
        if d in s:
            return clear(s.replace(d, ''))
    return s


def where_is_def(s):
    for d in def_phrases:
        idx = s.find(d)
        if idx != -1:
            break
    return idx, d


def is_definition(s):
    L = s.strip(punctuation).split()
    if len(L) == 0:
        return False
    last_word = L[-1]
    if last_word in SINGLE_WORD_PREPOSITIONS:
        return False

    idx, d = where_is_def(s)

    if idx == -1:
        return False
    # check for pronouns occuring after definition phrase
    if any(pr in s[idx + len(d):].split() for pr in FORBIDDEN_PRONOUNS):
        return False

    return True


def pad(s):
    return ' ' + s + ' '


def has_pronoun(s):
    return any(pad(x) in pad(s) for x in PRONOUNS)


def question_simplify(s):
    s = pad(s)
    for q in QUESTION_WORD:
        s = s.replace(pad(q), ' ')
    return s.strip()
