import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, BatchNormalization, Reshape

from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import scipy.io as sio

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# use data with 40 40 3 

def load_data():
    img = np.load('datasets/img_40_40.npy')
    pos = np.load('datasets/pos_2_21.npy')
    return img, pos

# load dataset
x, y = load_data()

input_dim = (40, 40, 3)
output_dim = (2,21)

print(x.shape, y.shape)

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
y_train = np.reshape(y_train, (*y_train.shape, 1))
y_test = np.reshape(y_test, (*y_test.shape, 1))
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


def create_model():
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

model = create_model()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

model.summary()

batch_size=64
epochs=100
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)


import time
import matplotlib.pyplot as plt

timestamp = int(time.time())

model.save(f'model/{timestamp}.h5')

plt.plot(history.history['mse'],'r')
plt.plot(history.history['val_mse'],'b')
plt.legend({'training mse':'r', 'validation mse': 'b'})
plt.savefig(f'image/history{timestamp}.png', dpi=300)
plt.show()
