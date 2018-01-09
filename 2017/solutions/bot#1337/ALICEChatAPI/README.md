aichat
=======

Uses the ALICE AIML to run a chatbot in Flask.

Inspired by/based on [Justin Huang's chatbot](https://github.com/jstnhuang/chatbot). Which in turn uses:

> [Flask](http://flask.pocoo.org/) is used to run a simple web server.

> The chatbot is based on the free [Alice AIML](https://code.google.com/p/aiml-en-us-foundation-alice/) set.


# Requirements

- python 3

# Installation

- pip install -r requirements.txt

# Running

- python ai.py

# Server

- input: user sentences array
- output: sentence


# Docker run example with built container

```
docker build -t sld3/alice_chat:0.1.0 .
docker run -p 3000:3000 -v $(pwd):/src sld3/alice_chat:0.1.0
```
