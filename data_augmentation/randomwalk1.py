import numpy as np
import torch
import random
import math
from random import choice
from PIL import Image
from time import time
#导入随机函数
class RandomWalk():
    '''生成一个随机漫步的类'''

    def __init__(self, x=[100], y=[100], num_points=5000, rbbox=None):
        '''初始化属性'''
        self.num_points = num_points
        # 默认次数为5000

        # 所有随机初始漫步源于(0,0)
        self.x_values = x
        self.y_values = y
        self.rbbox = rbbox

    def fill_walk(self):
        '''计算随机漫步包含的点'''
        # 不断漫步，直到列表达到指定长度

        while len(self.x_values) < self.num_points:
            # 决定前进方向及距离
            x_direction = choice([1, -1])
            x_distance = choice([0, 1])
            x_step = x_direction * x_distance

            y_direction = choice([1, -1])
            y_distance = choice([0, 1])
            y_step = y_direction * y_distance

            # # 拒绝原地踏步
            if x_step == 0 and y_step == 0:
                continue

            # 计算下一个点的位置
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            if next_x < (self.rbbox[0] + self.rbbox[2]) and next_x >= self.rbbox[0] and next_y < (
                    self.rbbox[1] + self.rbbox[3]) and next_y >= self.rbbox[1]:
                # 把位置添加到列表
                self.x_values.append(next_x)
                self.y_values.append(next_y)


class RWR(object):
    def __init__(self, p=0.5, al=0.01, ah=0.4, ns=0.9, pixel=[0.4914, 0.4822, 0.4465], bbox=None):
        self.p = p
        self.al = al
        self.ah = ah
        self.ns = ns
        # self.pixel = np.random.uniform(0, 1, (1, 3))
        self.pixel = pixel
        self.bbox = bbox
        # print(self.pixel)

    def __call__(self, img, *args, **kwargs):
        """
        :param img: tensor
        :param args: tensor
        :param kwargs:
        :return: tensor
        """
        if random.uniform(0, 1) > self.p:
            return img

        # Objection_wise_erasing
        if self.bbox !=None:

            # 获取指定区域边框信息
            bx, by = self.bbox[0], self.bbox[1]
            bw, bh = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
            area = bw * bh

            target_area = random.uniform(self.al, self.ah) * area
            num_ratio = random.uniform(self.ns, 1 / self.ns)

            # 生成点集 临近点个数
            for i in range(100):
                # 获取指定区域内随机漫步区域信息
                rbw = int(round(math.sqrt(target_area / num_ratio)))
                rbh = int(round(math.sqrt(target_area * num_ratio)))

                if rbw < self.bbox[2] and rbh < self.bbox[3]:
                    rbx, rby = random.randint(bx, bx+bw - rbw), random.randint(by, by+bh - rbh)
                    # 获取随机区域内的一个随机起始点
                    pos = np.random.randint((rbx, rby), (rbx + rbw, rby + rbh), (1, 2))
                    img = np.array(img)
                    rw = RandomWalk(x=[pos[0][0]], y=[pos[0][1]], num_points=rbw * rbh,
                                    rbbox=[rbx, rby, rbw, rbh])  # 括号中的次数为随机漫步次数
                    rw.fill_walk()
                    for i in range(0, len(rw.x_values)):
#                        img[:, rw.y_values[i], rw.x_values[i]] = np.random.uniform(0, 1, (1, 3))
                        img[:, rw.y_values[i], rw.x_values[i]] = np.array([0.4914, 0.4822, 0.4465])
                    img = torch.from_numpy(img)
                    return img
            return img

        # image_wise_erasing
        iw, ih = img.size(2), img.size(1)
        area = iw * ih
        target_area = random.uniform(self.al, self.ah) * area
        num_ratio = random.uniform(self.ns, 1 / self.ns)
        # 生成点集 临近点个数
        for i in range(100):

            rbw = int(round(math.sqrt(target_area / num_ratio)))
            rbh = int(round(math.sqrt(target_area * num_ratio)))

            if rbw < iw and rbh < ih:
                rbx, rby = random.randint(0, iw-rbw), random.randint(0, ih-rbh)
                pos = np.random.randint((rbx, rby), (rbx+rbw, rby+rbh), (1, 2))
                img = np.array(img)
                rw = RandomWalk(x=[pos[0][0]], y=[pos[0][1]], num_points=rbw * rbh,
                                rbbox=[rbx, rby, rbw, rbh])  # 括号中的次数为随机漫步次数
                rw.fill_walk()
                for i in range(0, len(rw.x_values)):
#                    img[:, rw.y_values[i], rw.x_values[i]] = np.random.uniform(0, 1, (1, 3))
                    img[:, rw.y_values[i], rw.x_values[i]] = np.array([0.4914, 0.4822, 0.4465])
                img = torch.from_numpy(img)
                return img
        return img

if __name__ == "__main__":
    start = time()
    bbox = [263, 211, 324, 339]
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
    print(end - start)
    img.show()
    # img.save("../data/test_data/bath1.jpg")