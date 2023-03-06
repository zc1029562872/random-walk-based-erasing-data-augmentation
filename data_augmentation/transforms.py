import random
import math
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import *
# from data_augmentation.randomwalk import RWR
from data_augmentation.randomwalk1 import RWR

class RandErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        """
        :param img: tensor
        :param args: tensor
        :param kwargs:
        :return: tensor
        """
        ih, iw = img.size()[1], img.size()[2]
        if random.uniform(0, 1) > self.probability:
            return img

        img = np.array(img)

        for attempt in range(100):
            # 指定区域
            area = ih * iw

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            # 随机生成的宽高
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < iw and h < ih:
                # h1 = random.randint(0, ih - h)
                # w1 = random.randint(0, iw - w)

                # 随机坐标
                h1 = random.randint(0, ih - h)
                w1 = random.randint(0, iw - w)
                
                img[:, h1:h1+h, w1:w1+w] = torch.from_numpy(np.random.rand(3, h, w))
                
                img = torch.from_numpy(img)
                return img
        img = torch.from_numpy(img)
        return img

if __name__ == "__main__":
    rand = RandErasing(probability=1)
    img = Image.open("../VOC/JPEGImages/000005.jpg")
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = rand(img)
    img = Image.fromarray(np.array(img).transpose(1, 2, 0), mode="RGB")
    img.show()
