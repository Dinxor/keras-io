import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import numpy as np
import os

characters = ['2','3','4','5','6','7','9','A','C','D','E','F','H','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z']

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None)

num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

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

if __name__ == '__main__':
    prediction_model = load_model('prediction_model.h5')
    good = 0
    all = 0
    for line in open('list.txt', 'r'):
        name = line[:-1]
        fullname = 'd:/Work/Python/capcha/captcha_images/' + name
        symbs = name[-8:-4]
        all +=1
        img = tf.io.read_file(fullname)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.transpose(img, perm=[2, 1, 0])

        pred = prediction_model.predict(img)
        cap = decode_batch_predictions(pred)[0]
        if cap == symbs:
            good +=1
        else:
            print(name, all, symbs, cap, good)
    print('%s from %s' % (good, all))
