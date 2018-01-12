#!/usr/bin/env python

from flask import Flask, request, jsonify
from flask import render_template
import ai
import uuid


app = Flask(__name__)


@app.route("/respond", methods=['POST'])
def respond():
    # bot.reset() not work
    user_sentences = request.json['sentences']
    response = "..."
    session_id = uuid.uuid4().hex
    print(user_sentences, session_id)
    for s in user_sentences:
        response = bot.respond(s, session_id).replace("\n", "")
    return jsonify({'message': response})

if __name__ == '__main__':
    bot = ai.Chatbot()
    bot.initialize("aiml-dir")
    app.run(debug=False, host='0.0.0.0', port=3000)
