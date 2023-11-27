import requests
import logging
import json

logging.basicConfig(
    filename = "../logpath/log.log",
    level = logging.DEBUG
)

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