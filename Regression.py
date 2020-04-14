import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, BatchNormalization

from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import scipy.io as sio

def load_data(end, start=2):
    img_array = np.empty((0, 450, 450, 3)).astype('float32')
    pos_array = np.empty((0, 42)).astype('float32')
    BASE_DIR = 'datasets/AFLW2000-3D/AFLW2000'

    for i in range(start, end+1):
        file_name = f'image{i:0>5}'
        img, pos = None, None

        # print(f'loading image {file_name}.jpg ...')
        img = cv2.imread(f'{BASE_DIR}/{file_name}.jpg')

        # print(f'loading data {file_name}.mat ...')
        try:
            pos = sio.loadmat(f'{BASE_DIR}/{file_name}.mat')
            pos = pos.get('pt2d')

        except FileNotFoundError:
            pos = None

        if img is None:
            continue
        elif pos is None:
            continue
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = np.reshape(img, (1, 450, 450, 3)).astype('float32')
            img = img / 255.0
            pos = np.reshape(pos, (1, 42)).astype('float32')

            img_array = np.append(img_array, img, axis=0)
            pos_array = np.append(pos_array, pos, axis=0)
            
            if (int(file_name[-5:]) // 1000) == 0: 
                print(f'loaded {file_name}')

    return img_array, pos_array

# load dataset
x, y = load_data(400)

input_dim = (450, 450, 3)
output_dim = (42)

print(x.shape, y.shape)

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# y_train = y_train.astype('float32')
# y_test = y_test.astype('float32')

print(x_train.shape, x_test.shape)


def create_model():
    model = Sequential()
    model.add(Convolution2D(16, (5,5), activation='relu', input_shape=(450,450,3), padding='same'))
    model.add(MaxPooling2D(pool_size=(5,5)))
    model.add(Convolution2D(32, (5,5), activation='relu'))
    model.add(Convolution2D(64, (5,5), activation='relu'))
    model.add(Convolution2D(128, (5,5), activation='relu'))
    model.add(Convolution2D(256, (5,5), activation='relu'))
    model.add(Flatten())
    # model.add(Dense(512))
    model.add(Dense(42))

    return model

model = create_model()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

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
