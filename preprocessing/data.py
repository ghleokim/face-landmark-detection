import tensorflow as tf
import cv2
import numpy as np
import scipy.io as sio

from sklearn.model_selection import train_test_split

res = 128


def load_data(end, start=2):
    img_array = np.empty((0, res, res, 3)).astype('float32')
    pos_array = np.empty((0, 2, 21)).astype('float32')
    BASE_DIR = '../datasets/AFLW2000-3D/AFLW2000'

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

            img = tf.image.resize( img, (res, res) )
            
            img = np.reshape(img, (1, res, res, 3)).astype('float32')
            img = img / 255.0

            try:
                pos = np.reshape(pos, (1, 2, 21)).astype('float32')
            except ValueError:
                print(pos.shape, pos, file_name)
                continue

            img_array = np.append(img_array, img, axis=0)
            pos_array = np.append(pos_array, pos, axis=0)
            
            if (int(file_name[-5:]) // 1000) == 0:
                print(f'loaded {file_name}')

    return img_array, pos_array


img, pos = load_data(4375)
pos_flat = np.reshape(pos, (len(pos), 42))

np.save(f'../datasets/img_{res}_{res}', img)
np.save('../datasets/pos_2_21', pos)
np.save('../datasets/pos_1_42', pos_flat)

x_train, x_test, y_train, y_test = train_test_split(img, pos, test_size=0.2, random_state=42)

np.save(f'../datasets/img_{res}_{res}_train', x_train)
np.save(f'../datasets/img_{res}_{res}_test', x_test)
np.save('../datasets/pos_2_21_train', y_train)
np.save('../datasets/pos_2_21_test', y_test)

y_train_flat = np.reshape(y_train, (len(y_train), 42))
y_test_flat = np.reshape(y_test, (len(y_test), 42))

np.save('../datasets/pos_1_42_train', y_train_flat)
np.save('../datasets/pos_1_42_test', y_test_flat)

