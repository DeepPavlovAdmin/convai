FROM python:3.6.1

WORKDIR /src

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('perluniprops'); nltk.download('nonbreaking_prefixes'); nltk.download('stopwords')"

CMD gunicorn -w 1 -b 0.0.0.0:3000 server:app
