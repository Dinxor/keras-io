import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

from flask import Flask, request
from gevent.pywsgi import WSGIServer

characters = ['2','3','4','5','6','7','9','A','C','D','E','F','H','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z']

prediction_model = load_model('prediction_model.h5')

char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(characters), num_oov_indices=0, mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:,:4]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def get_code(img):
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [50, 150])
    img = tf.transpose(img, perm=[2, 1, 0])
    pred = prediction_model.predict(img)
    rez = decode_batch_predictions(pred)[0]
    return rez

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
MAX_FILE_SIZE = 16 * 1024 + 1

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = request.headers
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_bytes = file.read(MAX_FILE_SIZE)
            try:
                rez = get_code(file_bytes)
            except:
                rez = 'TEST'
            if rez.find('[UNK]'):
                rez = rez.replace('[UNK]', '')
                if len(rez):
                    while len(rez) < 4:
                        rez = rez[0]+rez
                else:
                    rez = 'QWER'
            return rez
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
