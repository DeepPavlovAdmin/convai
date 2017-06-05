# Conversational Intelligence Challenge baseline solution

For now it is a Telegram bot, but it could easily be switched to our server by providing ```base_url``` parameter in telegram bot initialisation [here](https://github.com/MIPTDeepLearningLab/ConvAI-baseline/blob/master/bot_code/bot.py#L60). More information about parameters of telegram bot framework could be found [here](http://python-telegram-bot.readthedocs.io/en/latest/).

The solutaion is based on two papers:
* Neural Question Generation from Text: A Preliminary Study
https://arxiv.org/abs/1704.01792

* Bidirectional Attention Flow for Machine Comprehension
https://arxiv.org/abs/1611.01603

We are using forked repo of Allen AI2 bi-att-flow: https://github.com/allenai/bi-att-flow

## Requirements
* Docker ver. 17.03+:
    * Ubuntu: https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository
    * Mac: https://download.docker.com/mac/stable/Docker.dmg
    * Docker-compose ver. 1.13.0+: https://docs.docker.com/compose/install/
* Python 3
* ZeroMQ
* pyzmq dependencies:
    * Ubuntu: ```sudo apt-get install libzmq3-dev```
    * Mac: ```brew install zeromq --with-libpgm```

Python packages will be installed by ```setup.sh``` script.

## Setup
Run ```setup.sh```

Setup will download docker images, models and data files, so you have no need to download any of that by yourself.

## Bot
Simply run:
```python3 bot_code/bot.py```
