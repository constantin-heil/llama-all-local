from flask import Flask, request, jsonify
from model_embeddings import MODELNAME, mean_pooling, embeddingModel

app = Flask(__name__)

em = embeddingModel(MODELNAME, mean_pooling)

@app.route('/getembeddings', methods = ["POST"])
def get_embeddings():
    try:
        payload = request.get_json()
        if 'text' not in payload:
            return jsonify(
                error = 'field "text" not found in data'
                ), 400
        
        rawtext = payload['text']
        embeddings = str(em(rawtext).tolist())

        return jsonify(
            embedding_model = MODELNAME,
            embeddings = embeddings
        ), 200
    
    except Exception as err:
        return jsonify(
            error = str(err)
        ), 500
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0")
        
