import spacy
from pathlib import Path
from typing import List
import requests
from requests.exceptions import ConnectionError
import logging
import json

logging.basicConfig(
    filename = "../logpath/log.log",
    level = logging.DEBUG
)

class documentChunker:
    def __init__(self, 
                 input_document: Path,
                 chunk_len: int,
                 chunk_overlap: int,
                 skiplines: int,
                 chunk_sep: str = "."):
        self.chunk_len = chunk_len
        self.chunk_overlap = chunk_overlap
        self.skiplines = skiplines
        self.chunk_sep = chunk_sep

        self.init_list(input_document)

    def init_list(self, input_document) -> None:
        logging.debug("DOCUMENTCHUNKER: Populating internal sentences")
        input_document = Path(input_document) if not isinstance(input_document, Path) else input_document
        nlp = spacy.load("en_core_web_sm")

        with input_document.open('r') as fh:
            lines = [line.strip() for line in fh][self.skiplines:]

        
        fulltext = " ".join(lines)
        self.sentences = [sentence.strip() for sentence in fulltext.split(self.chunk_sep) if len(sentence) > 100]

    def __iter__(self):
        numchunks = len(self.sentences) // (self.chunk_len - self.chunk_overlap)
        for i in range(numchunks):
            start_ix = i*(self.chunk_len - self.chunk_overlap)
            end_ix = start_ix + self.chunk_len
            yield self.chunk_sep.join(self.sentences[start_ix:end_ix])

class embeddingRequester:
    def __init__(self,
                 hostname: str,
                 port: str):
        self.hostname, self.port = hostname, port

    def get_embeddings(self, 
                       text: List[str]) -> List[List[str]]:
        text = [text] if isinstance(text, str) else text
        
        embeddings = []
        
        for chunk in text:
            req = {
                "text": chunk
            }

            while True:
                try:
                    res = requests.post(
                        "http://" + self.hostname + ":" + self.port + "/getembeddings",
                        json = req,
                        headers = {'Content-Type': "application/json"}
                    )
                except ConnectionError:
                    continue

                break

            if res.status_code != 200:
                logging.error("EMBEDDINGREQUESTER: Could not get embeddings")

            output = json.loads(res.content)
            output['embeddings'] = json.loads(output['embeddings'])

            embeddings.extend(output['embeddings'])

        return embeddings


        

        


        