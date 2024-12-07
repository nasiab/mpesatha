from flask import Flask, request, jsonify
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load models
sentence_model = SentenceTransformer("lighteternal/stsb-xlm-r-greek-transfer")
#model_path1 ='./demo_model'
topic_model = BERTopic.load("./demo_model", embedding_model=sentence_model)

@app.route('/')
def home():
    return "Welcome to the BERTopic API. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        # Check if the file is present in the request
        if 'messageFile' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # Get the uploaded file
        file = request.files['messageFile']

        # Ensure the file is a text file
        if file.filename == '' or not file.filename.endswith('.txt'):
            return jsonify({'error': 'Invalid file format. Please upload a .txt file.'}), 400

        # Read the content of the file
        txt = file.read().decode('utf-8')
    
    
    # Encode the text
        txt_embedding = sentence_model.encode([txt], show_progress_bar=True)
    
    # Get the topics and probabilities
        topics, probabilities = topic_model.transform([txt], embeddings=txt_embedding)
    
    # Convert topics to a list of integers
        topics_list = [int(topic) for topic in topics]

    # Convert probabilities to a list of floats
        probabilities_list = [float(prob) for prob in np.array(probabilities).flatten()]
    
    # Return the results as JSON
        return jsonify({'topics': topics_list, 'probabilities': probabilities_list})
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
