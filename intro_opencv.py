import cv2
import numpy as np
import matplotlib.pyplot as plt


# load it in GRAYSCALE color mode...
#image = cv2.imread("scense.jpg", 0)
image = cv2.imread("scense.jpg")
#image = cv2.imread("lena.jpg")
cv2.imshow('Analytics Vidhya Computer Vision', image)
cv2.waitKey(0)

print(image.shape)
print("Image Height: %d Pixels"%(image.shape[0]))
print("Image Width: %d Pixels"%(image.shape[1]))
print("Number Of Color Channels: %d "%(image.shape[2]))

h, w, c = image.shape
print("Dimensions of the image is:nnHeight:", h, "pixelsnWidth:", w, "pixelsnNumber of Channels:", c)

print(type(image))
print(image.dtype)
print(image)


print("Eroding, Dilation")
# Creating kernel
kernel = np.ones((5, 5), np.uint8)
# Using cv2.erode() method 
image_erode = cv2.erode(image, kernel)

filename = 'image_erode1.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, image_erode)

cv2.imshow('Eroded image1', image_erode)
cv2.waitKey(0)

kernel2 = np.ones((3, 3), np.uint8)
image_erode2 = cv2.erode(image, kernel2, cv2.BORDER_REFLECT)

filename = 'image_erode2.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, image_erode2)

cv2.imshow('Eroded image2', image_erode2)
cv2.waitKey(0)

print("Dilation")
kernel3 = np.ones((5,5), np.uint8)
image_dilation = cv2.dilate(image, kernel, iterations=1)

filename = 'image_dilation.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, image_dilation)

cv2.imshow('Dilation image', image_dilation)
cv2.waitKey(0)

print("Creating a Border")
# Using cv2.copyMakeBorder() method
image_border1 = cv2.copyMakeBorder(image, 25, 25, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)

filename = 'image_border1.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, image_border1)

cv2.imshow('image borders', image_border1)
cv2.waitKey(0)

#making a mirrored border
image_border2 = cv2.copyMakeBorder(image, 250, 250, 250, 250, cv2.BORDER_REFLECT)
filename = 'image_border2.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, image_border2)
cv2.imshow('image mirror', image_border2)
cv2.waitKey(0)

#making a mirrored border
image_border3 = cv2.copyMakeBorder(image, 300, 250, 100, 50, cv2.BORDER_REFLECT)
filename = 'image_border3.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, image_border3)
cv2.imshow('image mirror2', image_border3)
cv2.waitKey(0)

# Apply log transform.
c = 255/(np.log(1 + np.max(image)))
log_transformed = c * np.log(1 + image)
# Specify the data type.
log_transformed = np.array(log_transformed, dtype = np.uint8)
cv2.imwrite('log_transformed.jpg', log_transformed)
cv2.imshow('log_transformed image', log_transformed)
cv2.waitKey(0)

#Linear Transformation
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
# Define parameters.
r1 = 70
s1 = 0
r2 = 140
s2 = 255
# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)
# Apply contrast stretching.
contrast_stretch = pixelVal_vec(image, r1, s1, r2, s2)
# Save edited image.
cv2.imwrite('contrast_stretch.jpg', contrast_stretch)
cv2.imshow('contrast_stretch image', contrast_stretch)
cv2.waitKey(0)



def extract_bit_plane(cd):
    #  extracting all bit one by one 
    # from 1st to 8th in variable 
    # from c1 to c8 respectively 
    c1 = np.mod(cd, 2)
    c2 = np.mod(np.floor(cd/2), 2)
    c3 = np.mod(np.floor(cd/4), 2)
    c4 = np.mod(np.floor(cd/8), 2)
    c5 = np.mod(np.floor(cd/16), 2)
    c6 = np.mod(np.floor(cd/32), 2)
    c7 = np.mod(np.floor(cd/64), 2)
    c8 = np.mod(np.floor(cd/128), 2)
    # combining image again to form equivalent to original grayscale image 
    cc = 2 * (2 * (2 * c8 + c7) + c6) # reconstructing image  with 3 most significant bit planes
    to_plot = [cd, c1, c2, c3, c4, c5, c6, c7, c8, cc]
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, i in zip(axes.flat, to_plot):
        ax.imshow(i, cmap='gray')
    plt.tight_layout()
    plt.show()
    return cc

print("extract_bit_plane")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('Gray Image', gray)
reconstructed_image = extract_bit_plane(gray)    
cv2.imshow('extract_bit_plane', reconstructed_image)


print("Constructing a small synthetic image")
con_img = np.zeros([256, 256])
con_img[0:32, :] = 40 # upper row
con_img[:, :32] = 40 #left column
con_img[:, 224:256] = 40 # right column
con_img[224:, :] = 40 # lower row
con_img[32:64, 32:224] = 80 # upper row
con_img[64:224, 32:64] = 80 # left column
con_img[64:224, 192:224] = 80 # right column
con_img[192:224, 32:224] = 80 # lower row
con_img[64:96, 64:192] = 160 # upper row
con_img[96:192, 64:96] = 160 # left column
con_img[96:192, 160:192] = 160 # right column
con_img[160:192, 64:192] = 160 # lower row
con_img[96:160, 96:160] = 220
plt.imshow(con_img)
plt.show()


resized_image = cv2.resize(src=image, dsize=(200, 200))

cv2.imshow('Analytics Vidhya Computer Vision - resized image', resized_image)
cv2.waitKey(0)

image_rotated_90_DEG_clockwise = cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Rotated 90 Degrees Clockwise', image_rotated_90_DEG_clockwise)
cv2.waitKey(0)

image_rotated_180_DEG_clockwise = cv2.rotate(src=image, rotateCode=cv2.ROTATE_180)
cv2.imshow('Rotated 180 Degrees Clockwise', image_rotated_180_DEG_clockwise)
cv2.waitKey(0)

image_blurred = cv2.blur(src=image, ksize=(5, 5))
cv2.imshow('Blurred image1', image_blurred)
cv2.waitKey(0)

image_blurred1 = cv2.blur(src=image, ksize=(75, 75))
cv2.imshow('Blurred image2', image_blurred1)
cv2.waitKey(0)

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
cv2.imshow('Sharpened image', image_sharp)
cv2.waitKey(0)

print("Watermark")
image = cv2.imread("scense.jpg")
hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
l_b=np.array([0,0,220])
u_b=np.array([255,255,255])
mask=cv2.inRange(hsv,l_b,u_b)
dst = cv2.inpaint(image,mask,5,cv2.INPAINT_TELEA)
#dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)
cv2.imshow('Watermark image', dst)
cv2.waitKey(0)


print("Removing Background Noise")
pts1  = np.float32([[57 , 49],[419 , 45],[414 , 477],[56 , 475]])
pts2 = np.float32([[0,0],[image.shape[0],0],[image.shape[0],image.shape[1]],[0,image.shape[1]]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(image, matrix, (512,512))
cv2.imshow("Removing Background Noise",result)
cv2.waitKey(0)

print("Denoising Colour Images")
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 15, 8, 8, 15)
# Save edited image.
cv2.imwrite('denoised_image.jpg', denoised_image)
cv2.imshow("denoised_image",denoised_image)
cv2.waitKey(0)

print("Analyze an image using Histogram")
histr = cv2.calcHist([image],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()

# alternative way to find histogram of an image
plt.hist(image.ravel(),256,[0,256])
plt.show()

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram = cv2.calcHist([grey_image], [0], None, [256], [0, 256])
plt.plot(histogram, color='k')
plt.show()


for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color = col)
    plt.xlim([0, 256])
plt.show()




print("Filter Images")
kernel = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])
kernel = kernel/9
res = cv2.filter2D(image, -1, kernel)
cv2.imshow("Filter image",res)
cv2.waitKey(0)

print("Convert Images into Cartoon")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),-1)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,10)
color = cv2.bilateralFilter(image, 20, 245, 245)
cartoon = cv2.bitwise_and(color, color, mask=edges)
cv2.imshow("cartoon image",cartoon)
cv2.waitKey(0)

print("Image Contrast")
plt.hist(image.ravel(), 256, [0,256], color='crimson')
plt.ylabel("Number Of Pixels", color='crimson')
plt.xlabel("Pixel Intensity- From 0-255", color='crimson')
plt.title("Histogram Showing Pixel Intensity And Corresponding Number Of Pixels", color='crimson')
plt.show()


cv2.destroyAllWindows()