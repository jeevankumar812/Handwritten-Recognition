import numpy as np
from tensorflow.keras.models import load_model
from segment import segment_characters

model = load_model("model/model.h5")

def predict_word(img_path):

    chars = segment_characters(img_path)
    result = ""

    # get numeric prediction
    for ch in chars:
        ch = ch/255.0
        ch = ch.reshape(1,64,64,1)
        pred = model.predict(ch)
        result += str(np.argmax(pred))

    # convert numbers to letters
    result = result.replace("1","a")
    result = result.replace("0","b")

    return result
