import tensorflow as tf

model_name = '1586755058'

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
num = 4375

data = sio.loadmat(f'datasets/AFLW2000-3D/AFLW2000/image{num:0>5}.mat')

print(data['pt2d'], data['pt2d'].shape)
print(data['Pose_Para'])
X, Y = data['pt2d'][0], data['pt2d'][1]

img = cv2.imread(f'datasets/AFLW2000-3D/AFLW2000/image{num:0>5}.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img_input = img / 255.0
img_input = np.reshape(img/255.0, (1, 450, 450, 3))
img_input = img_input.astype('float32')

pred = new_model.predict(img_input)
print(pred.shape)

pred_X, pred_Y = pred[0][:21], pred[0][21:]

print(pred_X.shape, pred_Y.shape)

print(pred_X, X)
print(pred_Y, Y)


print(img.shape)

plt.imshow(img)
plt.scatter(X,Y)
plt.scatter(pred_X, pred_Y)
plt.savefig(f'image/{model_name}.png', dpi=300)
plt.show()
