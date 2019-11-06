# coding=utf-8
from __future__ import division, print_function
import os
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from predict import *
from utils import *
from config import get_config
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

global classesM
classesM = ['N','V','L','R','Paced','A','F']#,'f','j','E','a','J','Q','e','S']

print('Check http://127.0.0.1:5000/')


def model_predict(img_path):
    data = uploadedData(img_path, csvbool = True)
    sr = data[0]

    data = data[1:]
    size = len(data)
    if size > 9001:
        size = 9001
        data = data[:size]
    div = size // 1000
    data, peaks = preprocess(data, config)
    return predictByPart(data, peaks)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        if not f:
            return "No file!"
        basepath = os.path.dirname(__file__)
        mkdir_recursive('uploads')
        
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        try:
            print("You already analyzed the data file. We delete it and re-analyze it!")
            os.remove(file_path)
        except:
            print("The file is new!")
        f.save(file_path)
        predicted, result = model_predict(file_path)
        length = len(predicted)

        result = str(length) +" parts of the divided data were estimated as the followings with paired probabilities. \n"+result
        
        return result
    return None


if __name__ == '__main__':
    config = get_config()
    #app.run(port=5001, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
