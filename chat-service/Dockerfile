FROM ubuntu:latest

COPY requirements.txt .
COPY server.py .

RUN mkdir /logpath
RUN mkdir /historydb

RUN apt update && apt install -y python3 git gcc python3-pip libpq-dev python3-dev
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install -r requirements.txt
RUN python -m pip install huggingface-cli