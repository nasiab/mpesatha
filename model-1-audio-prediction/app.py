from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from flask_cors import CORS
from predict_audio import predict_audio

app = Flask(__name__)
CORS(app)

#@app.route('/')
#def home():
 #   return 'hello'
#app = Flask(__name__, static_folder='static')

#@app.route('/')
#def home():
#    return send_from_directory('static', 'index.html')
@app.route('/')
def home():
    return "Service is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file:
        # Ensure the uploads directory exists
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        # Save the file to a temporary location
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        
        try:
            # Predict the class of the audio file
            predicted_class = predict_audio(file_path)
            
            # Remove the file after prediction
            os.remove(file_path)
            
            return jsonify({'predicted_class': int(predicted_class)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the uploads directory exists before starting the app
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    
    # Change the host to '0.0.0.0' to bind to all interfaces
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)                                                  