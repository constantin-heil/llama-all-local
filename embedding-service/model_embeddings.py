from transformers import AutoTokenizer, AutoModel
import torch
from typing import Callable, Union, Iterable
import argparse
from os import getenv

MODELNAME = getenv("MODELNAME", "sentence-transformers/all-MiniLM-L6-v2")

def get_args():
    ap = argparse.ArgumentParser(
        description = "Get embeddings for a string"
    )

    ap.add_argument(
        '-s', 
        '--string', 
        help = "Input string",
        required = True
        )
    
    return vars(ap.parse_args())

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min = 1e-9)

class embeddingModel:
    def __init__(self, modelname: str, agg_fun: Callable):
        self.modelname = modelname
        self.agg_fun = agg_fun
        self.initialize_model()
        
    def initialize_model(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(self.modelname)
        self.mod = AutoModel.from_pretrained(self.modelname)
        
    def __call__(self, inputs: Union[str, Iterable[str]]) -> torch.Tensor:
        inputs = [inputs] if isinstance(inputs, str) else inputs
        tokens = self.tok(inputs, padding = True, truncation = True, return_tensors = "pt")
        with torch.no_grad():
            raw_embeddings = self.mod(**tokens)
            
        agg_embeddings = self.agg_fun(raw_embeddings, tokens["attention_mask"])
        return agg_embeddings

if __name__ == "__main__":
    cmdargs = get_args()

    em = embeddingModel(MODELNAME, mean_pooling)
    outp = em(cmdargs['string'])

    print(outp)

    
    

