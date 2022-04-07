import base64
import re

import numpy as np
import tensorflow as tf
from flask import Flask, request
from keras.models import model_from_json
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import resize

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()

graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)

@app.route('/')
def index():
    return "Oops, nothing here!"

def stringToImage(img):
    imgstr = re.search(r'base64, (.*)', str(img)).group(1)
    with open('image.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict/', methods=['POST'])
def predict():
    global model, graph
    imgData = request.get_data()
    try:
        stringToImage(imgData)
    except:
        f = request.files['img']
        f.save('image.png')
    x = plt.imread('image.png')
    x = rgb2gray(rgba2rgb(x))
    x = resize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    print('predict')

    with graph.as_default():
        model = model_from_json(model_json)
        model.load_weights('weights.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])       
        prediction = model.predict(x)
        response = np.argmax(prediction, axis=1)
        return str(response[0])
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
