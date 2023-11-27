from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from pgvector.sqlalchemy import Vector
from sqlalchemy.exc import SQLAlchemyError
from parse_document import documentChunker, embeddingRequester
from os import getenv
import os
import logging

DB_USERNAME = getenv("DB_USERNAME", "root")
DB_PASSWORD = getenv("DB_PASSWORD", "password")
DB_HOST = getenv("DB_HOST", "localhost")
DB_PORT = getenv("DB_PORT", "5433")
EMBEDDING_DIMENSION = getenv("EMBEDDING_DIMENSION", 384)
EMBEDDINGSERVICE_HOST = getenv("EMBEDDINGSERVICE_HOST", "localhost")
EMBEDDINGSERVICE_PORT = getenv("EMBEDDINGSERVICE_PORT", "5000")
LOGPATH = getenv("LOGPATH", "../logpath")

logging.basicConfig(
    filename = os.path.join(LOGPATH, "log.log"),
    level = logging.DEBUG
)

class Base(DeclarativeBase):
    pass

class Embedding(Base):
    __tablename__ = "embeddings_table"

    id: Mapped[int] = mapped_column(primary_key = True)
    rawtext: Mapped[str]
    embedding = mapped_column(Vector(int(EMBEDDING_DIMENSION)))

    def __repr__(self):
        return f"Embedding(id={self.id}, rawtext={self.rawtext[:20]}..., embedding={self.embedding[:5]}...)"

if __name__ == "__main__":
    engine = create_engine(f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres")

    while True:
        try:
            engine.connect()
        except SQLAlchemyError as err:
            logging.warning(str(err))
            continue
        logging.warning("write_db.py: CONNECTED TO VECTORSTORE")
        break


    Base.metadata.create_all(engine)

    logging.debug("initializing objects")
    chunker = documentChunker(
        "documents/input.txt",
        chunk_len = 5,
        chunk_overlap = 2,
        skiplines = 200
    )

    requester = embeddingRequester(
        EMBEDDINGSERVICE_HOST,
        EMBEDDINGSERVICE_PORT
    )

    logging.debug("getting text chunks")
    chunks = [chunk for chunk in chunker]
    logging.debug(f"getting embeddings for {len(chunks)} chunks")
    embeddings = requester.get_embeddings(chunks)
    logging.debug("have embeddings")

    with Session(engine) as session:
        for c, e in zip(chunks, embeddings):
            session.add(Embedding(rawtext = c, embedding = e))

        logging.debug("writing to database")
        session.execute(text("CREATE INDEX ON embeddings_table USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000)"))
        session.commit()
        logging.debug("writing to database - DONE")