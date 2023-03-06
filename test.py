import numpy as np
from tqdm import tqdm
import random
import torch
import torchvision
import torch.utils.data as data
import os
from data_augmentation import transforms
import torchvision
from utils.utils import get_lr_scheduler
from utils.utils_fit import train
from PIL import Image

if __name__ == "__main__":
    # output = torch.tensor(np.random.randint(0, 100, (2, 100)))
    # values, pred = output.topk(5, 1, True, True)
    # target = torch.tensor(np.random.randint(0, 100, (2,)))
    # pred = pred.t()
    # target = target.view(1, -1).expand_as(pred)
    # print(target.shape)
    # print(output)
    # print(pred)


    # for epoch in range(10):
    #     pbar = tqdm(total=10, desc=f'Epoch {epoch + 1}/{10}', postfix=dict, mininterval=0.3)
    #     for i in range(10):
    #         pbar.set_postfix(**{
    #             'loss': (epoch + 1) / 10
    #         })
    #         pbar.update(1)
    #     pbar.close()

    transform_train = transforms.Compose({
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandErasing(),
    })
    dataloader = torchvision.datasets.CIFAR100
    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    print(len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print(targets)
        break

    # model_type = "resnet-20"
    # if not os.path.exists("./logs/" + model_type + "-logs"):
    #     os.mkdir("./logs/" + model_type + "-logs")

    # dataloader = torchvision.datasets.CIFAR100
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     transforms.RandErasing(),
    # ])
    #
    # trainset = dataloader(root='./data', train=True, download=False, transform=transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    #
    # for i, (inputs, targets) in enumerate(trainloader):
    #
    #     inputs, targets = inputs.cuda(0), targets.cuda(0)
    #     inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    #     print(inputs[0].shape)

    # a = torch.rand((5, 128))
    # b = a[:5].reshape(-1).float().sum(0)
    # print(b)

    # a = 4
    # torch.save(a, "./logs/m.txt")



