import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

model_name = '1586757470'

new_model = tf.keras.models.load_model(f'model/{model_name}.h5')

with open(f'model/{model_name}.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    new_model.summary(print_fn=lambda x: fh.write(x + '\n'))

import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# num = 2179
# num = 4375

# 2197 : test set

for i in range(2179, 2200):
    num = i

    try:
        data = sio.loadmat(f'datasets/AFLW2000-3D/AFLW2000/image{num:0>5}.mat')
    except FileNotFoundError:
        continue

    print(data['pt2d'], data['pt2d'].shape)
    print(data['Pose_Para'])
    X, Y = data['pt2d'][0], data['pt2d'][1]

    img = cv2.imread(f'datasets/AFLW2000-3D/AFLW2000/image{num:0>5}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img_input = img / 255.0
    img_input = tf.image.resize( img, (40,40) )

    img_input = np.reshape(img_input/255.0, (1, 40, 40, 3))
    img_input = img_input.astype('float32')

    pred = new_model.predict(img_input)
    print(pred.shape)

    pred_X, pred_Y = pred[0][0], pred[0][1]

    print(pred_X.shape, pred_Y.shape)

    print(pred_X, X)
    print(pred_Y, Y)


    print(img.shape)

    plt.title(f'image{num:0>5}.jpg')
    plt.imshow(img)
    plt.scatter(X,Y)
    plt.scatter(pred_X, pred_Y)
    plt.savefig(f'image/{model_name}.png', dpi=300)
    plt.show()
