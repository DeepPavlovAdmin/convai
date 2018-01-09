# ConvAI bot#1337

Skill-based Conversational Agent that took 1st place at 2017 NIPS Conversational Intelligence Challenge (http://convai.io).

**We still update our Conversational Agent and the latest version could be found in our repository**: https://github.com/sld/convai-bot-1337

Here is submitted to **ConvAI Finals version** of the Agent (on 12th November): https://github.com/sld/convai-bot-1337/tree/032d5f6f5cc127bb56d29f0f0c6bbc0487f98316

# Abstract

The chatbot developed for the ConvAI challenge. Our bot is
capable of conversing with humans about given text (e.g. a paragraph from
Wikipedia article). The conversation is enabled by a set of skills, including
chit-chat, topics detection, text summarization, question answering and question
generation. The system has been trained in a supervised fashion to select an
appropriate skill for generating a response. Furthermore, we have developed an
overall dialog quality scorer and next utterance scorer to correct agent's
policy. Our bot is implemented with open source software and open data; it is
self-hosted, and employs a supervised dialog manager with a linear hierarchy.
The latter allows a researcher to focus on skill implementation rather than
finite state machine development.

# Getting Started

For brief overview the bot#1337 take a look on next resources:

- [one-page abstract](https://www.researchgate.net/publication/322037222_Skill-based_Conversational_Agent)
- [presentation](https://www.researchgate.net/publication/322037067_Skill-based_Conversational_Agent)

## Prerequisites

- Docker version 17.05.0-ce+
- docker-compose version 1.13.0+
- Min. 4 Gb RAM + Swap (4 Gb), recommended 8 Gb RAM
- 2 Gb hard drive space
- Tested on Ubuntu 16.04

## Installing

Download and put trained models to folders:

```
./setup.sh
```

Build containers:

```
docker-compose -f docker-compose.yml -f telegram.yml build
```

Setup config.py, do not forget to put TELEGRAM token:

```
cp dialog_tracker/config.example.py dialog_tracker/config.py
```

dialog_tracker/config.py should look like this:

```
WAIT_TIME = 15
WAIT_TOO_LONG = 60
version = "17 (24.12.2017)"
telegram_token = "your telegram token"
```

## Running the bot

This command will run the telegram bot with your telegram token:

```
docker-compose -f docker-compose.yml -f telegram.yml up
```

# Running the tests

Run the bot by using json api server:

```
docker-compose -f docker-compose.yml -f json_api.yml up
```

Run the tests:

```
python dialog_tracker/tests/test_json_api.py http://0.0.0.0:5000
```

# Authors

- Idris Yusupov (http://github.com/sld)
- Yurii Kuratov (http://github.com/yurakuratov)

# License

This project is licensed under the GPLv3 License - see the LICENSE file for details.



