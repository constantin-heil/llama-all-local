# Fully local llama2 based model

This uses https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF and hosts a complete semantic search app including:
 - Vectorstore based on pgvector 
 - Embedding model 
 - Chat session memory (WIP)

 In order to set up clone the repo and run
 ```
 make up
 ```

 Send requests using to localhost:5001/chat
 ```
 curl -X POST -H 'Content-Type: application/json' -d '{"text": "hello"}' localhost:5001/chat
 ```