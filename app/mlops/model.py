from mlops.preprocess import preprocess
from tensorflow.keras.models import load_model

import numpy as np
import glob
#Loading the model


def load():
    model = load_model('Model_dp')
    return model


def predict(model , img):
    preprocessed = preprocess(img)
    return np.round(model.predict(preprocessed))[0][1]


if __name__ == '__main__':
    img = glob.glob('/mnt/h/dataset-part1/dick/1qrv06cp0v471.jpg')
    model = load()
    tab = ['dick' , 'no_dick']
    result = int(predict(model , img)[0][1])
    print('Model have predicted : ' + tab[result])
