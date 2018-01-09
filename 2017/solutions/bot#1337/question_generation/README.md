# Description

It is a question-generator model. It takes text and an answer as input
and outputs a question.

Question generator model trained in seq2seq setup by using http://opennmt.net.

# Environment

- Docker ver. 17.03+:

   - Ubuntu: https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository
   - Mac: https://download.docker.com/mac/stable/Docker.dmg

- Docker-compose ver. 1.13.0+: https://docs.docker.com/compose/install/
- Python 3
- pyzmq dependencies: Ubuntu `sudo apt-get install libzmq3-dev` or for Mac `brew install zeromq --with-libpgm`

# Setup

- run `./setup`.
This script downloads torch question generation model,
installs python requirements, pulls docker images and runs
opennmt and corenlp servers.


# Usage

`./get_qnas "<text>"` - takes as input text and outputs tsv.
- First column is a question,
- second column is an answer,
- third column is a score.

## Example

Input:

```
./get_qnas "Waiting had its world premiere at the \
  Dubai International Film Festival on 11 December 2015 to positive reviews \
  from critics. It was also screened at the closing gala of the London Asian \
  Film Festival, where Menon won the Best Director Award."
```

Output:

```
who won the best director award ? menon -2.38472032547
when was the location premiere ?  11 december 2015  -6.1178450584412
```


# Notes

- First model feeding may take a long time because of CoreNLP modules loading.
- Do not forget to install pyzmq dependencies.
