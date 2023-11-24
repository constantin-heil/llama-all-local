from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from pgvector.sqlalchemy import Vector
from parse_document import documentChunker, embeddingRequester
from os import getenv, file
import logging

DB_USERNAME = getenv("DB_USERNAME", "root")
DB_PASSWORD = getenv("DB_PASSWORD", "password")
DB_HOST = getenv("DB_HOST", "localhost")
DB_PORT = getenv("DB_PORT", "5433")
EMBEDDING_DIMENSION = getenv("EMBEDDING_DIMENSION", 384)
LOGPATH = getenv("LOGPATH", "../logpath")

logging.basicConfig(
    filename = os.file.path("LOGPATH", "dbwriting.log"),
    level = logging.DEBUG
)

class Base(DeclarativeBase):
    pass

class Embedding(Base):
    __tablename__ = "embeddings_table"

    id: Mapped[int] = mapped_column(primary_key = True)
    rawtext: Mapped[str]
    embedding = mapped_column(Vector(EMBEDDING_DIMENSION))

    def __repr__(self):
        return f"Embedding(id={self.id}, rawtext={self.rawtext[:20]}..., embedding={self.embedding[:5]}...)"

if __name__ == "__main__":
    engine = create_engine(f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres")
    Base.metadata.create_all(engine)

    logging.debug("initializing objects")
    chunker = documentChunker(
        "documents/input.txt",
        chunk_len = 5,
        chunk_overlap = 2,
        skiplines = 200
    )

    requester = embeddingRequester(
        "localhost",
        "5000"
    )

    logging.debug("getting text chunks")
    chunks = [chunk for chunk in chunker]
    logging.debug("getting embeddings")
    embeddings = requester.get_embeddings(chunks)
    logging.debug("have embeddings")

    with Session(engine) as session:
        for c, e in zip(chunks, embeddings):
            session.add(Embedding(rawtext = c, embedding = e))

        session.commit()
        







