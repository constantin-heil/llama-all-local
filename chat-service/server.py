from flask import Flask, request, jsonify
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from typing import List, Optional
from os import getenv
import os
import logging
import requests
import json
from llama_cpp import Llama
from functools import partial

DB_USERNAME = getenv("DB_USERNAME", "root")
DB_PASSWORD = getenv("DB_PASSWORD", "password")
DB_HOST = getenv("DB_HOST", "localhost")
DB_PORT = getenv("DB_PORT", "5433")
EMBEDDING_DIMENSION = getenv("EMBEDDING_DIMENSION", 384)
EMBEDDINGSERVICE_HOST = getenv("EMBEDDINGSERVICE_HOST", "localhost")
EMBEDDINGSERVICE_PORT = getenv("EMBEDDINGSERVICE_PORT", "5000")
LOGPATH = getenv("LOGPATH", "../logpath")
IS_TESTMODE = int(getenv("IS_TESTMODE", "0"))
MODELFILE = getenv("HG_FILE")

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

            if res.status_code != 200:
                logging.error("EMBEDDINGREQUESTER: Could not get embeddings")

            output = json.loads(res.content)
            output['embeddings'] = json.loads(output['embeddings'])

            embeddings.extend(output['embeddings'])

        return embeddings

def text_lookup(qtext: str,
                embeddingrequester: embeddingRequester,
                engine: sqlalchemy.engine,
                top_n: Optional[int] = 5):
    """
    Send a single text to embedding service and obtain top n matches
    """
    embs = embeddingrequester.get_embeddings(qtext)
    if IS_TESTMODE:
        embs = [emb[:5] for emb in embs]

    logging.debug(f"Got embeddings {embs} for {qtext}")
    with engine.connect() as con:
        res = con.execute(
            text("select rawtext from embeddings_table order by embedding <=> :qemb limit :topn ;"),
            parameters = {'qemb': str(embs[0]), 'topn': top_n}
        )

    reslist = ["- " + r[0] for r in res]
    bullet_list = "\n".join(reslist)
    return bullet_list

def get_prompt(userquery: str,
               semantic_list: str) -> str:
    imperative1 = "<|im_start|>system\nYou will answer the following question only if you know the answer, otherwise say you do not know\n"
    imperative2 = "You will put strong emphasis on the information in the following list, each starting with '-'\n"
    semantic_list = semantic_list + "\n<|im_end|>\n"
    userquery = "<|im_start|>user\n" + userquery + '\n<|im_start|>assistant'

    return imperative1 + imperative2 + semantic_list + userquery

def call_llm(llm, prompt: str, max_tokens: int, stop: List[str]) -> str:
    return llm(
        prompt, max_tokens, stop, echo = False
    )
    
llm = Llama(
    model_path = MODELFILE,
    n_ctx = 2048,
    n_threads = 4,
    n_gpu_layers = 0
)

mod = partial(call_llm, llm = llm, max_tokens = 512, stop = ["</s>"])

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
    logging.debug(f"Received TEXT: {inputtext}")
    bullet_list = text_lookup(
        inputtext,
        em,
        vectorstore_engine
        )
    logging.debug(f"Got lookup for: {inputtext}")
    final_prompt = get_prompt(
        userquery = inputtext,
        semantic_list = bullet_list
    )

    output = mod(prompt = final_prompt)
    return jsonify(
        response = output,
        input = inputtext, 
        fullprompt = final_prompt
    ), 200

if __name__ == "__main__":
    logging.debug("Running webserver")
    app.run(
        host = '0.0.0.0',
        port = 5001
    )
