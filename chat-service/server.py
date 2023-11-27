from flask import Flask, request, jsonify
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from typing import List
from os import getenv
import os
import logging
import requests
import json
from ctransformers import AutoModelForCausalLM

DB_USERNAME = getenv("DB_USERNAME", "root")
DB_PASSWORD = getenv("DB_PASSWORD", "password")
DB_HOST = getenv("DB_HOST", "localhost")
DB_PORT = getenv("DB_PORT", "5433")
EMBEDDING_DIMENSION = getenv("EMBEDDING_DIMENSION", 384)
EMBEDDINGSERVICE_HOST = getenv("EMBEDDINGSERVICE_HOST", "localhost")
EMBEDDINGSERVICE_PORT = getenv("EMBEDDINGSERVICE_PORT", "5000")
LOGPATH = getenv("LOGPATH", "../logpath")

MODELNAME = "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF"
FILENAME = "openhermes-2.5-mistral-7b.Q4_K_M.gguf"

logging.basicConfig(
    filename = os.path.join(LOGPATH, "log.log"),
    level = logging.DEBUG
)

class embeddingRequester:
    def __init__(self,
                 hostname: str,
                 port: str):
        self.hostname, self.port = hostname, port

    def get_embeddings(self, 
                       text: List[str]):
        text = [text] if isinstance(text, str) else text
        
        embeddings = []
        
        for chunk in text:
            req = {
                "text": chunk
            }

            res = requests.post(
                "http://" + self.hostname + ":" + self.port + "/getembeddings",
                json = req,
                headers = {'Content-Type': "application/json"}
            )

            if res.status_code != "200":
                logging.error("EMBEDDINGREQUESTER: Could not get embeddings")

            output = json.loads(res.content)
            output['embeddings'] = json.loads(output['embeddings'])

            embeddings.extend(output['embeddings'])

        return embeddings

def text_lookup(text: str,
                embeddingrequester: embeddingRequester,
                engine: sqlalchemy.engine,
                top_n: int = 5):
    """
    Send a single text to embedding service and obtain top n matches
    """
    embs = embeddingrequester.get_embeddings(text)
    with engine.connect() as con:
        res = con.execute(
            text("select rawtext from embeddings_table order by embedding <=> :qemb limit :topn ;"),
            parameters = {'qemb': str(embs[0]), 'topn': top_n}
        )

    reslist = ["- " + r[0] for r in res]
    bullet_list = "\n".join(reslist)
    return bullet_list

def get_prompt(userquery: str,
               semantic_list: str,
               systemmessage: str = "") -> str:
    imperative1 = "You will answer the following question only if you know the answer, otherwise say you do not know\n"
    imperative2 = "You will put strong emphasis on the information in the following list, each starting with '-'\n"
    systemmessage = systemmessage + '\n'
    userquery = userquery + '\n'

    return systemmessage + imperative1 + userquery + imperative2 + semantic_list
    
mod = AutoModelForCausalLM.from_pretrained(
    MODELNAME, 
    model_file = FILENAME,
    model_type = "mistral",
    gpu_layers = 0
    )

em = embeddingRequester(
    EMBEDDINGSERVICE_HOST,
    EMBEDDINGSERVICE_PORT
)

app = Flask(__name__)

#### Define a sqlite database to hold chat history
class Base(DeclarativeBase):
    pass

class History(Base):
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(primary_key = True)
    text: Mapped[str]
    date: Mapped[str]

    def __repr__(self):
        return f"History (id= {self.id}, text= {self.text}, date= {self.date})"
    
history_engine = create_engine(f"sqlite:///historydb/db.sqlite")
Base.metadata.create_all(history_engine)

vectorstore_engine = create_engine(f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres")

@app.route("/chat", methods = ["POST"])
def inputandresponse():
    payload = request.get_json()
    if "text" not in payload:
        return jsonify(
            error = "payload must contain text field"
        ), 500
    
    inputtext = payload["text"]
    logging.debug("Received TEXT: {inputtext}")
    bullet_list = text_lookup(
        inputtext,
        em,
        vectorstore_engine
        )
    logging.debug("Got lookup for: {inputtext}")
    final_prompt = get_prompt(
        userquery = inputtext,
        semantic_list = bullet_list
    )

    output = mod(final_prompt)
    return jsonify(
        response = output
    ), 200

if __name__ == "__main__":
    app.run(
        host = '0.0.0.0',
        port = 5001
    )