import csv
import random
from sys import argv


topic_map = {
    "topic_0": "politics",
    "topic_1": "nationalities",
    "topic_2": "military",
    "topic_3": "culture",
    "topic_4": "games",
    "topic_5": "biology",
    "topic_6": "engineering",
    "topic_7": "city",
    "topic_8": "sports",
    "topic_9": "research",
    "topic_10": "films",
    "topic_11": "contries",
    "topic_12": "business",
    "topic_13": "environment",
    "topic_14": "noise",
    "noise_0": "noise",
    "noise_1": "noise",
    "noise_2": "music"
}


templates = [
    "I think this text is connected with {}",
    "I think this text is connected with {}. I can ask you a question about it.",
    "Am I right that topic of the text is {}?",
    "I guess this text about {}. I can ask you a question about it.",
    "Let's talk about {}. I think it is familiar with the text.",
    "Let's talk about {}, or I can ask you a question about this text.",
    "What do you think about {}? Is it semantically near with the text?"
]


def generate_response(topic_with_score, only_good):
    if only_good is True and not is_good_topic(topic_with_score):
        return ""

    responses = generate_all_responses(topic)
    return random.sample(responses, k=1)[0]


def generate_all_responses(topic):
    return [t.format(topic) for t in templates]


def is_good_topic(topic_with_score):
    if topic_with_score[1] > 0.25:
        return True
    return False


def get_top3_topics(filename):
    with open(filename, 'r') as f:
        row = list(csv.DictReader(f, delimiter=';'))[0]

    sorted_items = sorted(row.items(), key=lambda x: x[1])
    top3 = list(reversed(sorted_items[-4:-1]))
    top3 = [(topic_map[k], float(v)) for k, v in top3 if topic_map[k] != 'noise']
    return top3


def print_top3_scores(filename):
    top3 = get_top3_topics(filename)
    for k, v in top3:
        print("{}\t{}\t{}".format(topic_map[k], k, v))


def print_good_response(filename):
    top3 = get_top3_topics(filename)
    top1 = top3[0]
    print(generate_response(top1, True))


def get_results(filename):
    top3 = get_top3_topics(filename)
    result = []
    for k, v in top3:
        responses = generate_all_responses(k)
        result.append({'topic': k, 'score': v, 'responses': responses})
    return result


if __name__ == '__main__':
    mode = argv[1]
    filename = argv[2]
    if mode == 'top3_scores':
        print_top3_scores(filename)
    elif mode == 'good_response':
        print_good_response(filename)
    elif mode == 'get_results':
        print(get_results(filename))

