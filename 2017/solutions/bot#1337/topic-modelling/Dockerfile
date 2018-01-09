FROM ofrei/bigartm:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('perluniprops'); nltk.download('nonbreaking_prefixes'); nltk.download('wordnet')"

WORKDIR /src

CMD gunicorn -w 4 -b 0.0.0.0:3000 server:app
