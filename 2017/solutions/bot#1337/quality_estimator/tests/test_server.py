import requests
import json
from sys import argv


def dialog_quality_test(url):
    data = {
        'thread': [
            {'text': 'Hi!', 'userId': 'bot'},
            {'text': 'Hey! How are you?', 'userId': 'user'}
        ]
    }

    r = requests.get(url, json=data)
    print(r.json())
    assert r.json()['quality_label'] in [0, 1, 2]


def utterance_quality_test(url):
    data = {
        'thread': [
            {'text': 'Hi!', 'userId': 'bot'},
            {'text': 'Hey! How are you?', 'userId': 'user'}
        ],
        'current': {'text': 'I am fine, thx.', 'userId': 'bot'}
    }

    r = requests.get(url, json=data)
    print(r.json())
    assert r.json()['quality_label'] in [1, 2]


if __name__ == '__main__':
    base_url = argv[1]
    dialog_quality_test(base_url + 'dialog_quality')
    utterance_quality_test(base_url + 'utterance_quality')
