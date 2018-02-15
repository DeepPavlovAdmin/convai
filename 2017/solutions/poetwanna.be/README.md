# poetwanna.be

A dialogue system (chatbot) made at the University of Wrocław.
Ranked 1st ex aequo in the NIPS 2017 Conversational Intelligence Challenge http://convai.io/ .

## Abstract

We present **poetwanna.be**, a chatbot submitted by the University of Wrocław
to the NIPS 2017 Conversational Intelligence Challenge.
As a dialogue system, it is able to conduct a conversation with a user
in a natural language.
It is tasked primarily with providing a context-aware question answering
(QA) functionality,
and secondarily maintaining the conversation by keeping the user engaged.

The chatbot is composed of a number of sub-modules, which independently prepare
replies to user’s prompts and assess their own confidence.
To answer questions, our dialogue system relies heavily on factual data
sourced mostly from Wikipedia and DBpedia,
data of real user interactions in public forums,
as well as data concerning general literature.
Where applicable, modules are trained on large datasets using GPUs.
However, to comply with the competition's requirements,
the final system is compact and runs on commodity hardware.

See also [workshop slides](../../nips17_wrkshp/poetwannabe.pdf).

## System Requirements

* 2 core CPU
* 16 GB RAM
* 50 GB HDD space
* (optionally) CUDA-enabled GPU

## Installation

Set up the `data/` directory containing models and corpora.
The script downloads `11 GB` of data, which unpacks to `30 GB`.
```
cd poetwanna.be
./download_data.sh
```
Build the Docker image.
```
docker build -t poetwannabe .
```
Run the image, mapping appriopriate directories.
```
docker run -it -v `pwd`:/qa_nips -v `pwd`/data:/data poetwannabe docker_bin/convai_cli
```
When using a GPU, run with `nvidia-docker`.
```
nvidia-docker run -it -v `pwd`:/qa_nips -v `pwd`/data:/data poetwannabe docker_bin/convai_cli
```

## Using the Telegram Interface (Optional)

To start the Telegram interface used during the competition
set `Bot ID` in the `chatbot/config.py` file.
```
# Convai
convai_bot_id = ''  # TODO Set ID
```
Finally run the container.
```
docker run -it -v `pwd`:/qa_nips -v `pwd`/data:/data poetwannabe docker_bin/convai_competition
```
If using a GPU:
```
nvidia-docker run -it -v `pwd`:/qa_nips -v `pwd`/data:/data poetwannabe docker_bin/convai_competition
```
## Training (Optional)

We supply pre-trained QA neural models in the `data/` directory.
Training procedure is described in the [QA model training guide](chatbot/talker/squad_talker/scripts/README.md).

## Data Sources

Word embeddings:
* GloVe word embeddings (https://nlp.stanford.edu/projects/glove)
* Word2Vec trained on Google News (https://code.google.com/archive/p/word2vec)

Question answering:
* English Wikipedia (https://en.wikipedia.org)
* Simple English Wikipedia (https://simple.wikipedia.org)
* SQuAD (https://rajpurkar.github.io/SQuAD-explorer)
* DBpedia (http://downloads.dbpedia.org/2016-04)
* unofficial DBpedia pagerank (http://people.aifb.kit.edu/ath/#DBpedia_PageRank)
* British English thesaurus (https://sourceforge.net/projects/brit-thesaurus)

Response matching:
* Wikiquote (https://en.wikiquote.org)
* chatterbot-corpus Python package (https://github.com/gunthercox/chatterbot-corpus)
* Tatoeba (https://tatoeba.org)
* dialogue corpus from the ConvAI pre-NIPS round (http://deephack.us15.list-manage.com/track/click?u=e2a9d77f36e424d716883fc18&id=bedde60fd7&e=4ff2819cc3)

Trivia:
* Moxquizz (http://moxquizz.de/download.html)
* Tat's Trivia (http://tatarize.nfshost.com)
* Irc-wiki (https://web.archive.org/web/20150323142257/http://irc-wiki.org)

## Authors (alphabetically)

* Jan Chorowski (https://github.com/janchorowski)
* Adrian Lancucki (https://github.com/alancucki)
* Szymon Malik (https://github.com/smalik169)
* Maciej Pawlikowski (https://github.com/maciek16180)
* Pawel Rychlikowski (https://github.com/PawelRychlikowski)
* Pawel Zykowski (https://github.com/zyks)
