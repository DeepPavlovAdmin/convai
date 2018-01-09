#!/usr/bin/env python

from flask import Flask, request, jsonify
from intent_classifier import IntentClassifier

app = Flask(__name__)
intent_classifier = IntentClassifier(
    path_to_datafile='./data/data.tsv',
    path_to_embedding='./data/glove.6B.100d.txt'
)

@app.route("/get_intent", methods=['POST'])
def respond():
    text = request.json['text']
    intent, score = _get_intent(text)
    return jsonify({'intent': intent, 'score': score})


def _get_intent(text):
    """
    Args:
        text: input utterance

    Returns:
        intent and its score

    """
    scores = intent_classifier.get_scores(text)
    max_intent, max_intent_score = intent_classifier.knn(text)
    print(scores, max_intent, max_intent_score)
    return max_intent, max_intent_score


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
