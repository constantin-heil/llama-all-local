# Local test bed for Retrieval-Augmented Generation

This is a complete RAG applicaiton that can use an artibrary GGUF quantized model hosted on Hugginface, as an example https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF 

This will build complete RAG app including:
 - Vectorstore based on pgvector 
 - Embedding model 
 - Locally run LLM

## Quickstart

Make sure you have docker installed and configured.

Then clone the rep and run:
```
make up
```

Send requests to the chat service by using the script requestsender.py.
```
python ./requestsender.py -i '<Your query here>' -f
```

## Configuration

The state of the branch as is is to use https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF.

This is determined in .env, by setting the variables ```HG_REPO``` and ```HG_FILE```.

Since different models use differently formatted prompts, create a new Jinja2 prompt template for your model if needed. Be sure to include the variables ```INPUTQUERY``` and ```BULLET_LIST```.

The templates are found under **chat-service/prompt_templates**.