import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



data = sio.loadmat('datasets/AFLW2000-3D/AFLW2000/image02179.mat')

print(data['pt2d'], data['pt2d'].shape)
print(data['Pose_Para'])
X, Y = data['pt2d'][0], data['pt2d'][1]

img = cv2.imread('datasets/AFLW2000-3D/AFLW2000/image02179.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)

plt.imshow(img)
plt.scatter(X,Y)

# r,g,b = cv2.split(img)

# r = r.flatten()
# g = g.flatten()
# b = b.flatten()

# ax.scatter(r,g,b)

plt.show()