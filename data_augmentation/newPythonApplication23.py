import numpy as np
import random
import cv2
import cv2 as cv
from PIL import Image
from data_augmentation.randomwalk import RandomWalk

srcImage = cv.imread('../VOC/JPEGImages/000005.jpg')
img = cv.imread('../VOC/JPEGImages/000005.jpg')
# img = Image.open('../VOC/JPEGImages/000005.jpg')
# scale_percent = 100 # 缩放尺寸
# shape = (512, 512)
# srcImage = cv.resize(srcImage, shape, interpolation = cv.INTER_AREA)
# srcImage = cv.cvtColor(srcImage, cv.COLOR_BGR2GRAY)
num_seed = 10


# img[211, 263: 324, :] = [255, 0, 0]
# img[339, 263: 324, :] = [255, 0, 0]
# img[211: 339, 263, :] = [255, 0, 0]
# img[211: 339, 324, :] = [255, 0, 0]

# pos = np.random.randint((0,0),(srcImage.shape[0],srcImage.shape[1]),(num_seed,2))    # image-wise RBR
pos = np.random.randint((263, 211), (324, 339), (num_seed, 2))

print(np.shape(img))

for i in range(num_seed):
    img[pos[i][1], pos[i][0], :] = [0, 0, 255]

output = srcImage.copy()

walknum = 500
# pixel = np.random.randint(0, 255, (3,))  # 填充的像素值
pixel = [0, 0, 255]

for a in range(0, num_seed):   #数量
    rw = RandomWalk([pos[a][0]], [pos[a][1]], walknum, minpos=(263, 211), maxpos=(324, 339))  # 括号中的次数为随机漫步次数
    # distance = random.randint(0, walknum)
    distance = walknum
    rw.fill_walk()

    for i in range(0, distance):
        if rw.x_values[i]<srcImage.shape[0] and rw.x_values[i]>0 and rw.y_values[i] <srcImage.shape[1] and rw.y_values[i]>0:
            output[rw.y_values[i], rw.x_values[i], :] = pixel


cv2.namedWindow('end',cv2.WINDOW_NORMAL)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv.imshow("end", output)
cv.imshow("img", img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()





