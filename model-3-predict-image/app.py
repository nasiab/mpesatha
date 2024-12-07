import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
#import librosa
import os, json, warnings
#from waitress import serve
#from mutagen import mp3, wave
from PIL import Image, ImageStat
#import imquality.brisque as brisque
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from image_captioning import feature_extractions, sample_caption
from datetime import datetime
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.preprocessing.text import tokenizer_from_json
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
#from collections import Counter
#from tinytag import TinyTag

app = Flask(__name__)
CORS(app)

model_image_classification = VGG16()

def image_captioning(filepath_alt=''):
    if request.method == 'POST' or filepath_alt !='':
        # prepare file
        if filepath_alt == '':
            file = request.files['messageFile']
            filepath = './temp/' + file.filename
            file.save(filepath)
        else:
            #
            filepath = filepath_alt

        # load image (tou malaka)
        # features = feature_extractions(filepath, model_image_classification)

        # load image (ours)
        image = load_img(filepath, target_size=(224, 224))

        if filepath_alt == '':
            os.remove(filepath)

    
        # produce catpion (ours)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model_image_classification.predict(image)
        t_label = decode_predictions(yhat)
        label = 'The following objects have been detected: '
        for item in t_label[0]:
            if item[2] > 0.1: label += item[1] + ', '
        label = label.replace("_", " ");
        label = label[:-2]

        # respond
        response = {
            "caption": label
        }
        # return f"""Image presents {label}."""
        return response

#Image-driven Bee-MATE
@app.route('/api/image/beemate', methods=['POST'])
def image_beemate(filepath_alt=''):
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    
    # Ensure the temp directory exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    if request.method == 'POST' or filepath_alt != '':
        
        file = request.files['messageFile']
        #filepath = './temp/' + file.filename
        filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath)
        image_caption = image_captioning(filepath)

        # load image (ours)
        img = load_img(filepath, target_size=(224, 224))

        image = img_to_array(img)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model_image_classification.predict(image)
        t_label = decode_predictions(yhat)
        detected_classes = []
        for item in t_label[0]:
            if item[2] > 0.1: detected_classes.append(item[1])
        #print (detected_classes)
        #nikos edits
        detected_pollution_classes = []

        
        pollution_source = []
        
        detected_pollution_classes = []

        filepath1 = os.path.join(os.path.dirname(__file__), 'beelexicon.txt')
        with open(filepath1, 'r') as file:
            # read all content of a file
            content = file.read()
            # check if string present in a file
            for image_class in detected_classes:
                print (image_class)
                if image_class in content:
                    audio_pollution_class = "Pollution source detected!"
                    detected_pollution_classes.append(image_class)
                    pollution_source = detected_pollution_classes
            
        if not pollution_source: pollution_source=["none"]
        response = {**image_caption, "pollution_source": pollution_source}
        print("Caller IP: " + request.remote_addr)
        print("Image service: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " ", response)

        # delete file
        os.remove(filepath)
        return response
    
if __name__ == '__main__':
    # Ensure the uploads directory exists before starting the app
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    
    # Change the host to '0.0.0.0' to bind to all interfaces
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)