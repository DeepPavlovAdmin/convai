# RLLChatBot

Conversational AI Agent submitted by McGill University, which was finalist in NIPS Conversational AI Challenge (http://convai.io)

The latest version of our work is in this repository: https://github.com/mike-n-7/convai, we are still working on updates.

## Abstract

The task of this challenge is to build a bot that can hold a conversation with humans about a given news article. We divide our approach into two high level steps:

- Generation of candidate responses based on news article and dialogue history.
We have an ensemble of generative models, retrieval models, and rule based models to generate those responses.
- Controlling of the conversation flow. At each turn, the generated candidate responses from each model are fed into a neural network that scores the responses based on their estimated human score

## Getting Started

### Prerequisites

- Docker (If you want to run in a docker environment)
- Min 32GB of RAM
- Min 50GB of disk space
- Tested on Ubuntu 16.04

### Installation

- Setup the models : run `./setup` in `models/` directory
- Download the data: run `./setup` in `data/` directory
- Install dependencies `pip install -r requirements.txt`
- Run the model by the script `./start`

## Running Docker

- After installing docker, build the image from this directory using the following command: `docker build -t convai .`
- Docker will create a virtual container with all the dependencies needed.
- Docker will autostart the bot whenever the container is run: `docker run convai`

## Running from DockerHub

Alternatively, you can run our model directly from DockerHub by pulling our repository:

```
docker pull ksinha/rllchatbot:nips
```


## File description

- **bot_q.py** : Main entry point of the chat bot, message selection logic can be implemented here.
- **models/** : Folder where model code is stored
- **data/** : Folder where data files are stored
- **config.py** : Configuration script, which has the location of data files as well as bot tokens. Replace the bot token with your ones to test.
- **models/wrappper.py** - Wrapper function which calls the models. Must implement `get_response` def.
- **models/setup** - shell script to download the models
- **data/setup** - shell script to download the data files and saved model files
- **model_selection_q.py** - Selection logic for best answer

## Bugs

Feel free to open an issue or submit a PR to https://github.com/mike-n-7/convai.

## Authors

McGill RLLDialog Team

- Koustuv Sinha (https://github.com/koustuvsinha)
- Nicolas Angelard-Gontier (https://github.com/NicolasAG)
- Peter Henderson (https://github.com/BreakEnd)
- Prasanna Parthasarathy (https://github.com/pparth03)
- Mike Noseworthy (https://github.com/mike-n-7)
- Joelle Pineau

