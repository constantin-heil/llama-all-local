FROM postgres:16

COPY setup.sql /docker-entrypoint-initdb.d

RUN apt update && apt install -y postgresql-16-pgvector