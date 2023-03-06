import numpy as np
import torch
import random
import math
from random import choice
from PIL import Image
from time import time

class RandomWalk(object):

    def __init__(self, x=[], y=[], num_points=None, w=None, h=None, minpos=None, maxpos=None):
        self.num_points = num_points
        self.x_values = x
        self.y_values = y
        self.w = w
        self.h = h
        self.minpos = minpos
        self.maxpos = maxpos

    def fill_walk(self):
        while len(self.x_values) < self.num_points:
            x_direction = choice([1, -1])
            x_distance = choice([0, 0, 0, 0, 1])
            x_step = x_direction * x_distance

            y_direction = choice([1, -1])
            y_distance = choice([0, 0, 0, 0, 1])
            y_step = y_direction * y_distance

            if x_step == 0 and y_step == 0:
                continue

            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            if self.minpos != None:
                if next_x < self.maxpos[0] and next_x > self.minpos[0] and next_y < self.maxpos[1] \
                        and next_y > self.minpos[1]:
                    # 把位置添加到列表
                    self.x_values.append(next_x)
                    self.y_values.append(next_y)
            else:
                if next_x < self.w and next_x >= 0 and next_y < self.h and next_y >= 0:
                    # 把位置添加到列表
                    self.x_values.append(next_x)
                    self.y_values.append(next_y)


class RWR(object):
    def __init__(self, p=0.5, al=0.01, ah=0.4, ns=1, pixel=[0.4914, 0.4822, 0.4465], bbox=None):
        self.p = p
        self.al = al
        self.ah = ah
        self.ns = ns
        self.pixel = np.random.uniform(0, 1, (1, 3))
        self.bbox = bbox
        print(self.pixel)

    def __call__(self, img, *args, **kwargs):
        """
        :param img: tensor
        :param args: tensor
        :param kwargs:
        :return: tensor
        """
        if random.uniform(0, 1) > self.p:
            return img

        iw, ih = img.size(2), img.size(1)
        area = iw * ih
        target_area = random.uniform(self.al, self.ah) * area
        num_ratio = random.uniform(self.ns, 1 / self.ns)

        # 生成点集 临近点个数
        ps = int(round(math.sqrt(target_area / num_ratio)))
        pn = int(round(math.sqrt(target_area * num_ratio)))
        pos = np.random.randint((0, 0), (iw, ih), (ps, 2))

        img = np.array(img)
        # for a in range(0, ps):  # 数量
        for a in range(0, ps):  # 数量
            rw = RandomWalk(x=[pos[a][0]], y=[pos[a][1]], num_points=pn, w=iw, h=ih)  # 括号中的次数为随机漫步次数
            rw.fill_walk()
            for i in range(0, pn):
                img[:, rw.y_values[i], rw.x_values[i]] = self.pixel

        img = torch.from_numpy(img)
        return img

if __name__ == "__main__":
    start = time()
    # bbox = [263, 211, 324, 339]
    img = Image.open("../VOC/JPEGImages/000005_1.jpg")
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1)
    print(img.size())
    # rw = RWR(p=1, ah=0.4, ns=1, bbox=bbox)
    rw = RWR(p=1, ah=0.4, ns=1)
    img = rw(img).permute(1, 2, 0)
    img = np.array(img)
    # img[bbox[1], bbox[0]: bbox[2], :] = [255, 0, 0]
    # img[bbox[3], bbox[0]: bbox[2], :] = [255, 0, 0]
    # img[bbox[1]: bbox[3], bbox[0], :] = [255, 0, 0]
    # img[bbox[1]: bbox[3], bbox[2], :] = [255, 0, 0]
    img = Image.fromarray(img)
    end = time()
    print(end-start)
    img.show()
    # img.save("../data/test_data/bath.jpg")
