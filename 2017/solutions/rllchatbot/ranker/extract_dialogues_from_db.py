import json
import pymongo
import argparse
import cPickle as pkl
import time
from collections import defaultdict
import copy


score_map = {0: 0,
             1: -1,
             2: +1}


def is_regular_alphabet(string):
    return all(ord(c) < 128 for c in string)


def query_db():
    """
    Collect messages from db.local (client side) and db.dialogs (server side)
    Builds a list of conversation with evaluation fields (from db.dialogs)
      and model & policy fields for each message in each conversation.
    - Loop through conversations in the client side (db.local)
    - Get the corresponding server's dialog and add to the list we want to return
    - Use the client's dialog to get model & policy fields of each convo msg
    :return: array of dictionaries. each dictionary is a conversation.
    """
    PORT = 8091
    CLIENT = '132.206.3.23'

    client = pymongo.MongoClient(CLIENT, PORT)
    db = client.convai

    chats = []

    # loop through each conversation on the client side because it only has valid
    #   conversations. db.dialogs also contains old and invalid convos.
    for d_local in list(db.local.find({})):
        d_id = d_local['dialogId']  # convo ID
        # get the same conversation from the server side (db.dialogs)
        d_servr = list(db.dialogs.find({'dialogId': d_id}))
        if len(d_servr) > 1:
            print "Error: two dialogs with same id (%s)!" % d_id
            continue
        elif len(d_servr) < 1:
            print "Warning: no dialog found in db.dialogs for dialogId %s." % d_id
            continue
        d_servr = d_servr[0]
        data = copy.deepcopy(d_servr)  # make hard copy of convo

        # map from local msg text to local msg object
        local_msgs = dict(
            [(msg['text'], msg) for msg in d_local['logs'] if msg['text'] is not None]
        )
        # list of messages in the server's convo: the order we want to keep
        servr_msgs = [msg for msg in d_servr['thread'] if msg['text'] is not None]

        # for each message in the server's dialog (the one we want to keep)
        for msg in servr_msgs:
            text = msg['text']
            if text not in local_msgs:
                print "Warning: msg (`%s`) from dialogId (%s) not found in db.local" % (msg['text'], d_id)
                model = 'none'
                policy = -1
            else:
                model = local_msgs[text].get('model_name', 'human_from_db')
                policy = local_msgs[text].get('policyID', -1)
            msg['model'] = model
            msg['policy'] = policy

        data['thread'] = servr_msgs
        chats.append(data)

    return chats

def valid_chat(usr_turns, bot_turns, k=2):
    # Check that user sent at least k messages and bot replied with 2 more messages
    long_enough = len(usr_turns) >= k and len(bot_turns) >= k

    # print "bot:%d %% usr:%d" % (len(bot_turns), len(usr_turns))
    # valid_flow = len(bot_turns) == len(usr_turns) + 2  # normal flow: bot - bot - (usr - bot)*n
    # early_stop = len(bot_turns) == len(usr_turns) + 1  # user sent /end before the bot reply or sent two msg in a row during the conversation
    # usr_sent_more_than_1msg == len(bot_turns) <= len(usr_turns)  # usr sent more than 1 message before the bot had time to reply

    ## TODO: Check for bad language
    polite = True
    
    ## Check that user voted at least 95% of all bot messages
    # novote = filter(lambda turn: turn['evaluation']==0, bot_turns)
    # voted = float(len(novote)) / len(bot_turns) < 0.15  # voted at least 95% of all bot turns
    voted = True

    return long_enough and polite and voted


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
        # get the user id for this chat if there is one
        uid = None
        for usr in dialog['users']:
            if usr['userType'] == 'ai.ipavlov.communication.TelegramChat':
               uid = usr['id']
        # skip that conversation if no user found
        if uid is None:
            print "Error: No user in this chat (%s), skipping it!" % dialog['dialogId']
            continue

        # get user_turns and bot_turns
        usr_turns = [msg for msg in dialog['thread'] if msg['userId'] == uid]
        bot_turns = [msg for msg in dialog['thread'] if msg['userId'] != uid]

        if valid_chat(usr_turns, bot_turns, k=4):
            # get article text for that conversation
            article = dialog['context'].strip().lower()
            # get full evaluation for that conversation
            full_eval = []
            for evl in dialog['evaluation']:
                if evl['userId'] == uid:
                    full_eval.append( (2.0*evl['quality'] + 1.0*evl['breadth'] + 1.0*evl['engagement']) / 4.0 )
            if len(full_eval) < 1:
                print "Error: no evaluation found for this conversation (%s), skipping it" % dialog['dialogId']
                continue
            elif len(full_eval) > 1:
                print "Error: more than one evaluation found for this conversation (%s), skipping it" % dialog['dialogId']
                continue
            full_eval = full_eval[0]

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

                # if the human talked
                if msg['userId'] == uid:
                    c = copy.deepcopy(context)
                    # human spoke twice in a row:
                    if last_sender_id == msg['userId']:
                        context[-1] = c[-1]+' '+msg['text'].strip().lower()
                    else:
                        context.append(msg['text'].strip().lower())
                        last_sender_id = msg['userId']

                # if the bot talked
                else:
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
                                'R': full_eval,
                                'policy': m['policy'],
                                'model': m['model']
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
                                'R': full_eval,
                                'policy': m['policy'],
                                'model': m['model']
                            })
                            added_instances_from_this_chat = True
                        # add bot response to context now
                        context.append(m['text'].strip().lower())
                        last_sender_id = m['userId']

    if voted_only:  # filter out messages here again
        # sometimes msg[r] is still 0 because msg[candidate] is composed of two msgs:
        #  one with +1, the other with -1, summing to 0
        formated_data = filter(lambda msg: msg['r'] != 0, formated_data)

    return formated_data


def main():
    parser = argparse.ArgumentParser(description='Create pickle data for training, testing ranker neural net')
    parser.add_argument('--voted_only', action='store_true', help='consider only voted messages')
    args = parser.parse_args()
    print args

    print "\nGet conversations from database..."
    json_data = query_db()
    print "Got %d dialogues" % len(json_data)

    # print '\n', json_data[0]

    # extract array of dictionaries of the form {'article':<str>, 'context':<list of str>, 'candidate':<str>, 'r':<-1,0,1>, 'R':<0-5>}
    print "\nReformat dialogues into list of training examples..."
    full_data = reformat(json_data, args.voted_only)
    print "Got %d examples" % len(full_data)

    # print '\n', json.dumps(full_data[:5], indent=4, sort_keys=True)

    print "\nSaving to json file..."
    file_prefix = "voted" if args.voted_only else "full"
    with open('./data/%s_data_db_%s.json' % (file_prefix, str(time.time())), 'wb') as handle:
        json.dump(full_data, handle)
    print "done."


if __name__ == '__main__':
    main()

