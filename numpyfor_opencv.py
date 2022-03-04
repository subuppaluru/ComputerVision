import numpy as np
import cv2

numpy_zeros = np.zeros(shape=(6, 6))
print("numpy_zeros")
print("=============")
print(numpy_zeros)
print("=============")

numpy_ones = np.ones(shape=(4, 4))
print("numpy_ones")
print("=============")
print(numpy_ones)
print("=============")

numpy_zeros[1:5, 1:5] = numpy_ones
print("numpy_zeros")
print("=============")
print(numpy_zeros)
print("=============")

print("Back to open cv")
image = cv2.imread("scense.jpg")

print("Actual Image Details")
print('Shape Of image: ', image.shape)
print("Rows of Pixels: %d Rows"%(image.shape[0]))
print("Columns of Pixels: %d Columns"%(image.shape[1]))
print("Color Channels: %d Color Channels"%(image.shape[2]))


sc = image[200:250, 400:450]
cv2.imshow('scense', image)
cv2.waitKey(0)

print("sc Image Details")
print('Shape Of sc: ', sc.shape)
print("Rows of Pixels: %d Rows"%(sc.shape[0]))
print("Columns of Pixels: %d Columns"%(sc.shape[1]))
print("Color Channels: %d Color Channels"%(sc.shape[2]))

cv2.imshow('scense1', sc)
cv2.waitKey(0)

sc = image[200:250, 400:450]

image[50:100, 15:65] = sc 
image[50:100, 75:125] = sc[:, ::-1, :] 
image[50:100, 150:200] = sc[:, ::-1, :] 
image[50:100, 225:275] = sc[:, :, :] 
image[50:100, 300:350] = sc[:, ::-1, :]  

cv2.imshow('Cropped Trees', image)
cv2.waitKey(0)


cv2.destroyAllWindows()