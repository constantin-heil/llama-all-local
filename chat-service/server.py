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
import jinja2
from llama_cpp import Llama
from functools import partial

DB_USERNAME = getenv("DB_USERNAME", "root")
DB_PASSWORD = getenv("DB_PASSWORD", "password")
DB_HOST = getenv("DB_HOST", "localhost")
DB_PORT = getenv("DB_PORT", "5433")
EMBEDDING_DIMENSION = getenv("EMBEDDING_DIMENSION", 384)
EMBEDDINGSERVICE_HOST = getenv("EMBEDDINGSERVICE_HOST", "localhost")
EMBEDDINGSERVICE_PORT = getenv("EMBEDDINGSERVICE_PORT", "5000")
LOGPATH = getenv("LOGPATH", "/logpath")
IS_TESTMODE = int(getenv("IS_TESTMODE", "0"))
MODELFILE = getenv("HG_FILE")
PROMPT_TEMPLATE = getenv("PROMPT_TEMPLATE", "prompt_templates/phi-2-gguf.txt")

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
    embs = embeddingrequester.get_embeddings(qtext)[0]

    logging.debug(f"Got embeddings {embs} for {qtext}")
    with engine.connect() as con:
        res = con.execute(
            text("select rawtext from embeddings_table order by 1 - (embedding <=> :qemb) desc limit :topn ;"),
            parameters = {'qemb': str(embs), 'topn': top_n}
        )

    logging.debug(f"Got lookup {res}")
    reslist = ["\n- " + r[0] for r in res]
    bullet_list = "\n".join(reslist)
    return bullet_list

def get_prompt(userquery: str,
               bullet_list: str) -> str:
    envpath, templatefn = PROMPT_TEMPLATE.split("/")
    env = jinja2.Environment(loader = jinja2.FileSystemLoader(envpath))
    template = env.get_template(templatefn)

    return template.render(
        USERQUERY = userquery, 
        BULLET_LIST = bullet_list
        )
    

def call_llm(llm, prompt: str, max_tokens: int, stop: List[str]) -> str:
    return llm(
        prompt, max_tokens, stop, echo = False
    )

logging.debug("Starting CHAT SERVICE")

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
    
    logging.debug(f"Got bullet_list {bullet_list}")
    final_prompt = get_prompt(
        userquery = inputtext,
        bullet_list = bullet_list
        )

    logging.debug(f"Using final prompt: {final_prompt}")
    output = llm(
        final_prompt,
        max_tokens = 512, 
        stop = ["</s>"],
        echo = False)['choices'][0]['text']
    
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
