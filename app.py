import webview
import threading
import h5py, json
import keras.backend as K
import numpy as np
import sys
import PIL
import PIL.ImageOps
import io, os, time, base64
import tensorflow as tf
import keras
try:
   wd = sys._MEIPASS
except AttributeError:
   wd = os.getcwd()
file_path = os.path.join(wd,'model.h5',"assets/index.html", "model.json" )

def initialize():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights('model.h5')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model._make_predict_function()
    return model


class Api:

    def __init__(self):
        return None

    def predict(self, params):
        tic = time.time()
        im = PIL.Image.open(io.BytesIO(base64.b64decode(params[22:])))
        im = im.convert("L")
        im = PIL.ImageOps.invert(im)
        im = im.resize((28, 28))
        greyscale_map = np.array(im)
        greyscale_map = greyscale_map.reshape(1, 1, 28, 28, order="A")
        greyscale_map = tf.keras.utils.normalize(greyscale_map, axis=1)
        predictions = model.predict(greyscale_map)
        toc = time.time()
        response = {
            'message': f"{np.argmax(predictions[0])}",
            'time': f"{round((toc-tic), 3)}",
            'amax': f"{np.amax(predictions[0]*100)}"
        }
        return response


if __name__ == '__main__':
    t = threading.Thread(target=initialize)
    t.daemon = True
    t.start()
    api = Api()
    webview.create_window('Deepcalculator', "assets/index.html", js_api=api,
                          debug=True, width=272, height=520, resizable=False)
