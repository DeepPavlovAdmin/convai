FROM python:3.6.1

WORKDIR /src

RUN apt-get update && apt-get install -y libzmq3-dev git

COPY requirements.txt /tmp/requirements-dt.txt
COPY from_question_generation/requirements.txt /tmp/requirements-fqg.txt
COPY from_factoid_question_answerer/requirements.txt /tmp/requirements-ffqa.txt
RUN pip install -r /tmp/requirements-dt.txt
RUN pip install -r /tmp/requirements-fqg.txt
RUN pip install -r /tmp/requirements-ffqa.txt

RUN git clone https://github.com/facebookresearch/fastText.git /fasttext && cd /fasttext \
  && git checkout 8f036268097c76a284ee98e63d9a17e2feffe715

RUN cd /fasttext && make
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('perluniprops'); nltk.download('nonbreaking_prefixes'); nltk.download('stopwords')"

CMD python main.py
