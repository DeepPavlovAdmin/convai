#!/usr/bin/env python3

import sys
import json
from jinja2 import Template
import codecs

from settings import bot_names, bot_type_names


def dialog_min_len(thread):
    dialog = dict()
    for t in thread:
        if t['userId'] not in dialog:
            dialog[t['userId']] = 0
        dialog[t['userId']] += 1
    return 0 if len(dialog.values()) == 0 else min(dialog.values())

def calc_score(q):
    if len(q) > 0:
        return round(sum(q) / float(len(q)), 3)
    else:
        return 0

def bot_leaderboard():
    bot_evaluations = dict()

    lines = sys.stdin.readlines()
    for line in lines:
        d = json.loads(line)
        if dialog_min_len(d['thread']) <= 2:
            continue

        user1_type = d['users'][0]['userType'] 
        user2_type = d['users'][1]['userType'] 
        if user1_type in bot_type_names:
            bot_id = d['users'][0]['id']
        elif user2_type in bot_type_names:
            bot_id = d['users'][1]['id']
        else:
            # dialogue between humans
            continue

        if bot_id not in bot_names:
            print('unknown bot {}'.format(bot_id), file=sys.stderr)
            continue
        bot_name = bot_names[bot_id]

        if bot_name not in bot_evaluations:
            bot_evaluations[bot_name] = []

        for e in d['evaluation']:
            if e['userId'] != bot_id and e['quality'] != 0:
                bot_evaluations[bot_name].append(e['quality'])

    leaderboard = []
    for bot in bot_evaluations:
        leaderboard.append((bot, calc_score(bot_evaluations[bot])))

    leaderboard.sort(key=lambda tup: tup[1], reverse=True)

    return leaderboard

with codecs.open('../index.md.template', encoding='utf-8', mode='r') as template:
    t = Template(template.read())

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stdout.write(t.render(board=bot_leaderboard()))

