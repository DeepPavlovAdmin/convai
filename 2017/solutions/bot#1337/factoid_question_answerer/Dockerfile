FROM python:3.5.0

RUN apt-get update && apt-get install -y git

ENV TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl
RUN pip3 install --upgrade $TF_BINARY_URL

RUN git clone https://github.com/MIPTDeepLearningLab/bi-att-flow /root/bi-att-flow && \
  cd /root/bi-att-flow && git checkout convai-demo

RUN cd /root/bi-att-flow && git pull

RUN pip3 install -r /root/bi-att-flow/requirements.txt
RUN python3 -c "import nltk; nltk.download('punkt')"

EXPOSE 1995

CMD cd /root/bi-att-flow && python3 run-demo.py
