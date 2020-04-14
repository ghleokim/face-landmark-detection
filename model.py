import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, BatchNormalization

def model_A():
    model = Sequential()
    model.add(Convolution2D(16, (5,5), activation='relu', input_shape=(450,450,3), padding='same'))
    model.add(MaxPooling2D(pool_size=(5,5)))
    model.add(Convolution2D(32, (5,5), activation='relu'))
    model.add(Convolution2D(64, (5,5), activation='relu'))
    model.add(Convolution2D(128, (5,5), activation='relu'))
    model.add(Convolution2D(256, (5,5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(42))

    return model

def model_40_40():
    """
        model input dimension 40x40x3
        model output dimension 2x21
    """
    model = Sequential()
    model.add(Convolution2D(8, (3,3), activation='relu', input_shape=(40,40,3), padding='same'))
    model.add(Convolution2D(16, (3,3), activation='relu'))
    model.add(Convolution2D(32, (3,3), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Convolution2D(128, (3,3), activation='relu'))
    model.add(Convolution2D(256, (3,3), activation='relu'))
    model.add(Convolution2D(42, (30,30), activation='relu'))
    model.add(Reshape((2, 21)))

    return model


def create_model_C():
    """
    model built from code of
    https://github.com/yinguobing/cnn-facial-landmark/blob/master/model.py
    """

    model = Sequential()
    model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(128,128,3), padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(128, (3,3), activation='relu'))
    model.add(Convolution2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(256, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(42))
    
    return model