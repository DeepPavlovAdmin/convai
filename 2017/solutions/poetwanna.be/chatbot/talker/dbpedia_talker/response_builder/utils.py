import re
from ..dbpedia.resource_finder import ResourceFinder

CAMELCASE_SPLIT_REGEX = re.compile('[a-zA-Z][^A-Z]*')


def get_plural(values):
    return 's' if len(values) > 1 else ''


def get_verb(values):
    return 'is' if len(values) == 1 else 'are'


def values_to_phrase(values):
    return ', '.join(map(strip_value, values))


def camelcase_to_phrase(camelcase_string):
    words = CAMELCASE_SPLIT_REGEX.findall(camelcase_string)
    return ' '.join(map(lambda s: s.lower(), words))


def strip_value(value):
    if value.startswith('"'):
        return value.split('"^^')[0].lstrip('"')
    if value.startswith('<http://dbpedia.org/resource/'):
        resource_name = value[len('<http://dbpedia.org/resource/'):].rstrip('>')
        return ' '.join(resource_name.split('_'))
    if value.endswith('"@en'):
        return value[:-len('"@en')].lstrip('"')
    if '_' in value:
        return ' '.join(value.split('_'))
    return value


def remove_nested_comments(text):
    ret, skip1c, skip2c = '', 0, 0
    for i in text:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')' and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret


def get_top_pagerank_values(values):
    rf = ResourceFinder()
    return sorted(values, key=lambda v: rf.get_resource_pagerank(v), reverse=True)[:3]
