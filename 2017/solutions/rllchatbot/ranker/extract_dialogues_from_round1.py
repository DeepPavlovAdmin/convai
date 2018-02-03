import json
import cPickle as pkl
import time
import copy
from collections import defaultdict
import argparse


score_map = {0: 0,
             1: -1,
             2: +1}


def is_regular_alphabet(string):
    return all(ord(c) < 128 for c in string)


def valid_chat(turns, k=2):
    # map from user id to the number of message sent
    user_utt = defaultdict(int)
    for msg in turns:
        user_utt[msg['userId']] += 1

    if len(user_utt) < 2:  # make sure the two users spoke
        return False

    for u in user_utt:  # for each user
        if user_utt[u] < k:  # check that it sent at least k messages
            return False

    return True


def reformat(json_data, voted_only=False):
    """
    Create a list of dictionaries of the form {'article':<str>, 'context':<list of str>, 'candidate':<str>, 'r':<-1,0,1>, 'R':<0-5>}
    TODO: make sure the list is ordered by article!! ie: [article1, ..., article1, article2, ..., article2, article3, ..., ..., article_n]
    :param json_data: list of conversations. each conversation is a dictionary
    :param voted_only: consider only the messages which have been up- or down- voted
    :return: list of training instances. each instance is a dictionary
    """
    formated_data = []

    for dialog in json_data:
        # get the bot id for this chat if there is one
        bid = None
        for usr in dialog['users']:
            if usr['userType'] == 'Bot':
               bid = usr['id']
        
        both_human = (bid is None)

        if valid_chat(dialog['thread'], k=2):
            # print json.dumps(dialog, indent=4, sort_keys=True)
            # get article text for that conversation
            article = dialog['context'].strip().lower()
            # get evaluations for that conversation from all humans involved
            full_evals = defaultdict(float)
            for evl in dialog['evaluation']:
                full_evals[evl['userId']] = (2.0*evl['quality'] + 1.0*evl['breadth'] + 1.0*evl['engagement']) / 4.0
            if len(full_evals) == 0:
                print "Warning: no full evaluation found for this conversation, skipping it"
                continue

            # Go through conversation to create a list of (article, context, candidate, score, reward, policy, model) instances
            context = []
            last_sender_id = None
            added_instances_from_this_chat = False  # True as soon as we add an instance
            for msg in dialog['thread']:
                # skip empty messages or messages written in non-unicode characters
                if len(msg['text'].strip().split()) == 0 or not is_regular_alphabet(msg['text'].strip().lower()):
                    continue

                # if begining of the converesation, just fill in the context
                if len(context) == 0:
                    context.append(msg['text'].strip().lower())
                    last_sender_id = msg['userId']

                # if both human: create an instance for all messages and add each message to context
                elif both_human:
                    # same human spoke twice in a row:
                    if last_sender_id == msg['userId']:
                        c = copy.deepcopy(context)
                        m = copy.deepcopy(msg)
                        prev_candidate = c[-1]  # save previous turn
                        context = c[:-1]        # remove last turn from context
                        if added_instances_from_this_chat:  # replace last instance by most recent
                            r_prev = formated_data[-1]['r']
                            r_new = min(max(r_prev + score_map[int(m['evaluation'])], -1), 1)  # sum evaluations of the two msg [-1,+1]
                            formated_data[-1] = {
                                'article': article,
                                'context': copy.deepcopy(context),
                                'candidate': prev_candidate+' '+m['text'].strip().lower(),  # include last turn in this turn
                                'r': r_new,
                                'R': full_evals[m['userId']],
                                'policy': -1,
                                'model': 'human_from_round1'
                            }
                        # add human response to context now
                        context.append(prev_candidate+' '+m['text'].strip().lower())
                    # other human replied to this human:
                    else:
                        c = copy.deepcopy(context)
                        m = copy.deepcopy(msg)
                        if (not voted_only) or (voted_only and score_map[int(m['evaluation'])] != 0):
                            # create new instance
                            formated_data.append({
                                'article': article,
                                'context': c,
                                'candidate': m['text'].strip().lower(),
                                'r': score_map[int(m['evaluation'])],
                                'R': full_evals[m['userId']],
                                'policy': -1,
                                'model': 'human_from_round1'
                            })
                            added_instances_from_this_chat = True
                        # add human response to context now
                        context.append(m['text'].strip().lower())
                        last_sender_id = m['userId']

                # if the bot talked
                elif msg['userId'] == bid:
                    # the bot spoke twice in a row:
                    if last_sender_id == msg['userId']:
                        c = copy.deepcopy(context)
                        m = copy.deepcopy(msg)
                        prev_candidate = c[-1]  # save previous turn
                        context = c[:-1]        # remove last turn from context
                        if added_instances_from_this_chat:  # replace last instance by most recent
                            r_prev = formated_data[-1]['r']
                            r_new = min(max(r_prev + score_map[int(m['evaluation'])], -1), 1)  # sum evaluations of the two msg [-1,+1]
                            formated_data[-1] = {
                                'article': article,
                                'context': copy.deepcopy(context),
                                'candidate': prev_candidate+' '+m['text'].strip().lower(),  # include last turn in this turn
                                'r': r_new,
                                'R': full_evals[m['userId']],
                                'policy': -1,
                                'model': 'bot_from_round1'
                            }
                        # add bot response to context now
                        context.append(prev_candidate+' '+m['text'].strip().lower())
                    # bot replied to human:
                    else:
                        c = copy.deepcopy(context)
                        m = copy.deepcopy(msg)
                        if (not voted_only) or (voted_only and score_map[int(m['evaluation'])] != 0):
                            # create new instance
                            formated_data.append({
                                'article': article,
                                'context': c,
                                'candidate': m['text'].strip().lower(),
                                'r': score_map[int(m['evaluation'])],
                                'R': full_evals[m['userId']],
                                'policy': -1,
                                'model': 'bot_from_round1'
                            })
                            added_instances_from_this_chat = True
                        # add bot response to context now
                        context.append(m['text'].strip().lower())
                        last_sender_id = m['userId']

                # if the (lonly) human talked
                else:
                    c = copy.deepcopy(context)
                    # same human spoke twice in a row:
                    if last_sender_id == msg['userId']:
                        context[-1] = c[-1]+' '+msg['text'].strip().lower()
                    else:
                        context.append(msg['text'].strip().lower())
                        last_sender_id = msg['userId']

                # print json.dumps(formated_data, indent=4, sort_keys=True)

    if voted_only:  # filter out messages here again
        # sometimes msg[r] is still 0 because msg[candidate] is composed of two msgs:
        #  one with +1, the other with -1, summing to 0
        formated_data = filter(lambda msg: msg['r'] != 0, formated_data)

    return formated_data


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--voted_only', action='store_true', help='only consider messages which has been voted')
    args = parser.parse_args()
    print args

    json_file = "/home/ml/nangel3/research/data/convai/round1.json"
    print "\nGet json data from %s..." % json_file
    with open(json_file, 'r') as handle:
        json_data = json.load(handle)
    print "Got %d dialogues" % len(json_data)

    # extract array of dictionaries of the form {'article':<str>, 'context':<list of str>, 'candidate':<str>, 'r':<-1,0,1>, 'R':<0-5>}
    print "\nReformat dialogues into list of training examples..."
    full_data = reformat(json_data, args.voted_only)
    print "Got %d examples" % len(full_data)

    # print json.dumps(full_data[4:20], indent=4, sort_keys=True)

    print "\nSaving to json file..."
    file_prefix = "voted" if args.voted_only else "full"
    with open('./data/%s_data_round1_%s.json' % (file_prefix, str(time.time())), 'wb') as handle:
        json.dump(full_data, handle)
    print "done."


if __name__ == '__main__':
    main()

