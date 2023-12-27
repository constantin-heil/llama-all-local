FROM ubuntu:latest

ARG HG_REPO
ENV HG_REPO=${HG_REPO}

ARG HG_FILE
ENV HG_FILE=${HG_FILE}

COPY requirements.txt .
COPY server.py .

RUN mkdir /logpath
RUN mkdir /historydb

RUN apt update && apt install -y python3 git gcc python3-pip libpq-dev python3-dev
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install -r requirements.txt
RUN python -m pip install huggingface-cli
RUN huggingface-cli download ${HG_REPO} ${HG_FILE} --local-dir .

CMD python server.py