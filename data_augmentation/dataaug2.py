# basic数据增强
import numpy as np 
import random
from PIL import Image, ImageEnhance
import math
#传进来是图片  而不是 array
#随机改变亮度
def random_brightness(img,lower=0.5,upper=1.5):
    e = np.random.uniform(lower,upper)
    img = ImageEnhance.Brightness(img).enhance(e)
    return img

#随机改变对比度
def random_contrast(img,lower=0.5,upper=1.5):
    e = np.random.uniform(lower,upper)
    img = ImageEnhance.Contrast(img).enhance(e)
    return img

#随机改变颜色
def random_color(img,lower=0.5,upper=1.5):
    e = np.random.uniform(lower,upper)
    img = ImageEnhance.Color(img).enhance(e)
    return img

#随机改变清晰度
def random_sharpness(img,lower=0.5,upper=1.5):
    e = np.random.uniform(lower,upper)
    img = ImageEnhance.Sharpness(img).enhance(e)
    return img

#随机等比例裁剪
def random_crop(img,max_ratio = 1.5):
    img = np.asarray(img)
    h,w,_ = img.shape 
    m = random.uniform(1,max_ratio)
    n = random.uniform(1,max_ratio)
    x1 = w *(1-1/m)/2
    y1 = h *(1-1/n)/2
    x2 = x1 + w * 1/m
    y2 = y1 + h * 1/n
    img = Image.fromarray(img)
    img = img.crop([x1,y1,x2,y2])

    type = [Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.ANTIALIAS]
    img = img.resize((w,h),type[random.randint(0,3)])
    return img

def random_flip(img,thresh = 0.5):
    img  = np.asarray(img)
    if random.random() >thresh:
        img  = img[:,::-1,:]  #左右翻转
    else :
        img = img[::-1,:,:]  #上下翻转
    img  = Image.fromarray(img)
    return img

#随机旋转
def random_rotate(img,thresh= 0.5):
    # 任意角度
    angle = random.randint(0,360)
    img = img.rotate(angle)
    return  img

#随机添加高斯噪声
def random_noise(img,max_sigma =50):
    img = np.asarray(img)
    sigma = random.uniform(0,max_sigma)
    noise = np.round(np.random.rand(img.shape[0],img.shape[1],3)*sigma).astype('uint8')
    img = img + noise
    img[img>255] = 255
    img[img<0] = 0
    img = Image.fromarray(img)
    return img
#将方法集合起来
def basic_augment(img):
    ops = [random_brightness, random_contrast, random_color, random_sharpness, random_crop, \
            random_flip, random_rotate, random_noise]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = ops[3](img)
    img = ops[4](img)
    img = ops[5](img)
    img = ops[6](img)
    img = ops[7](img)
    img = np.asarray(img)

    return img
# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread('../VOC/JPEGImages/000005.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.figure()
# img = basic_augment(img)
# plt.imshow(img)
# plt.show()
"""
区域删除技术
random erasing  cutout  gridmask
"""

# random erasing
def random_erasing(img, pro=0.5,sl = 0.02,sh=0.4,r1= 0.3,mean=[0.4914, 0.4822, 0.4465]):
    """
    pro： 擦除概率
    sl: min erasing area  擦除区域
    sh: max erasing area
    r1: min aspect ratio   长宽比
    mean: erasing value   擦除区域填充  随机填充
    """
    img = np.array(img)

    if np.random.rand() > pro:
        img = Image.fromarray(img)
        return img
    for i in  range(100):  #直到找到合适区域为止
        h,w,c =  img.shape
        area = h * w
        target_area  = np.random.uniform(sl,sh) * area
        aspect_ratio = np.random.uniform(r1,1/r1) 

        hh = int(round(math.sqrt(target_area * aspect_ratio)))  #round 四舍五入
        ww = int(round(math.sqrt(target_area / aspect_ratio)))

        if hh < h and ww<w:
            x1 = np.random.randint(0,w-ww)
            y1 = np.random.randint(0,h-hh)
            img[y1:y1+hh,x1:x1+ww,0] = mean[0]*255
            img[y1:y1+hh,x1:x1+ww,1] = mean[1]*255
            img[y1:y1+hh,x1:x1+ww,2] = mean[2]*255
            img = Image.fromarray(np.uint8(img))
            return img
    img = Image.fromarray(np.uint8(img))
    img = img.assary(img)
    return img
#cutout
def cutout(img,n_holes=3,length=10):
    """
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.
    """
    n_holes = np.random.randint(2,6)
    img = np.asarray(img)
    h,w,_ = img.shape
    mask = np.ones((h,w),dtype=np.float32)

    for i in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length//2,0,h)
        y2 = np.clip(y + length//2,0,h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1:y2,x1:x2] = 0.

    mask = np.expand_dims(mask, -1)
    mask = np.tile(mask, [1, 1, 3])
    img = img * mask
    img = Image.fromarray(np.uint8(img))
    img = img.asarry(img)
    return img

#Grid mask
def gridmask(img,d1=10,d2=50,rotate=90,ratio=0.5,mode=1,st_prob=0.7,current_epoch=90,EPOCH_NUM =100):
    prob = st_prob * min(1,current_epoch/EPOCH_NUM)
    if np.random.rand()> prob:
        img = np.asarray(img)
        img = Image.fromarray(img)
        return img
    
    img = np.asarray(img)
    h,w,_ = img.shape
    hh= int(1.5*h)
    ww = int(1.5*w)
   
    #d为一个mask单元的长度 （黑加白）
    d = np.random.randint(d1,d2)

    # l为（黑色）删除区域的边长
    l = math.ceil(d*ratio)

    mask = np.ones((hh, ww), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(hh//d):
        s = d*i +st_h
        t = min(s + l, hh)
        mask[s:t,:] *=0
    for i in range(hh//d):
        s = d*i + st_w
        t = min(s + l, ww)
        mask[:,s:t] *= 0
    #mask旋转角度
    r =  np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask) 
    mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w].astype(np.float32)
    if mode == 1:
        mask = 1 - mask
    mask = np.expand_dims(mask, -1)
    mask = np.tile(mask, [1, 1, 3])
    img = img * mask
    img = Image.fromarray(np.uint8(img))
    img = img.asarry(img)
    return img
#图像的混合
#samplePairing
def samplePairing(img1,img2):
    h = max(img1.shape[0],img2.shape[0])
    w = max(img1.shape[1],img2.shape[1])
    img = np.zeros((h,w,img1.shape[2]),'float32')
    img[:img1.shape[0],:img1.shape[1],:] = img1.astype('float32') /2
    img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32')/2
    if img1.dtype == 'uint8':
        return img.astype('uint8')   # 处理、返回归一化前图片 
    else:
        return img                   # 处理、返回归一化后图片
#mixup
def mixup(img1,img2,lambd):
    h = max(img1.shape[0],img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])
    img = np.zeros((h,w,img1.shape[2]),'float32')
    img[:img1.shape[0],:img1.shape[1]] = img1.astype('float32')*lambd
    img[:img2.shape[0],:img2.shape[1]] +=img2.astype('float32')*(1-lambd)
    if img1.dtype == 'uint8':
        return img.astype('uint8')   # 处理、返回归一化前图片 
    else:
        return img                   # 处理、返回归一化后图片


if __name__ == "__main__":
    img = Image.open("../VOC/JPEGImages/000005.jpg")
    img1 = random_erasing(img)
    img1.show()

