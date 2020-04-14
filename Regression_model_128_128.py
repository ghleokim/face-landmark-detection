import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, BatchNormalization, Reshape
from tensorflow.keras import callbacks

from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import scipy.io as sio

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# use data with 128 128 3 

def load_data():
    img = np.load('datasets/img_128_128.npy')
    pos = np.load('datasets/pos_1_42.npy')
    return img, pos

# load dataset
x, y = load_data()

input_dim = (128, 128, 3)
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

from model import create_model_C

model = create_model_C()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

model.summary()


# training section

# define callback to save history
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs.keys())
        print(callbacks.History())
        history = logs['history']

        plt.plot(history.history['mse'],'r')
        plt.plot(history.history['val_mse'],'b')
        plt.legend({'training mse':'r', 'validation mse': 'b'})
        plt.show()

batch_size=16
epochs=1000
history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[CustomCallback()]
                    )
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
