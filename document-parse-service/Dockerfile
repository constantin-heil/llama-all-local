FROM ubuntu:latest

COPY parse_document.py .
COPY write_db.py .
COPY requirements.txt .
COPY ./documents /documents

RUN mkdir /logpath

RUN apt update && apt install -y python3 git gcc python3-pip libpq-dev python3-dev
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install -r requirements.txt

CMD python3 write_db.py