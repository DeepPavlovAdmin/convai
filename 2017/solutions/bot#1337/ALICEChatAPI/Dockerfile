FROM python:3.6.1

WORKDIR /src
COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

CMD python server.py
