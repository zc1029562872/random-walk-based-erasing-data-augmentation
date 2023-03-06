import torchvision
import torch
import random
import numpy as np
import torch.nn as nn
import os
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from data_augmentation import transforms
from nets.Pre_Act_resnet import preresnet, preresnet18
from nets.resnet import resnet
from nets.wrn import wrn
from utils.utils import get_lr_scheduler, set_optimizer_lr, Logger, save_checkpoint, get_lr
from utils.utils_fit import train, test

if __name__ == "__main__":
    #------------------------------------#
    #    Cuda
    #------------------------------------#
    Cuda = True
    #------------------------------------#
    #    设置模型类型
    #    model type: pre-act-resnet, resnet, wrn
    #------------------------------------#
    model_type = "pre-act-resnet"
    #------------------------------------#
    #    权值文件
    #------------------------------------#
    model_pth = ""
    #------------------------------------#
    #    类别个数
    #------------------------------------#
    num_classes = 100
    #------------------------------------#
    #    设置世代数
    #------------------------------------#
    Init_epoch = 0
    End_epoch = 100
    #------------------------------------#
    #    设置batch_size
    #------------------------------------#
    train_batch_size = 128
    test_batch_size = 100
    #------------------------------------#
    #    设置学习率lr
    #    设置学习率schedule
    #------------------------------------#
    init_lr = 0.1
    momentum = 0.9
    gamma = 0.1
    weight_decay = 5e-4
    # schedule = [150, 225]
    schedule = [50, 75]
    #-----------------------------------------------------#
    #    如果有GPU，设置num_workers >= 2
    #-----------------------------------------------------#
    num_workers = 2
    #----------------------------------#
    #    模型保存间隔
    #----------------------------------#
    save_step = 5
    #------------------------------------#
    #    数据获取
    #------------------------------------#
    dataloader = torchvision.datasets.CIFAR100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RWR(p=0.3, al=0.01, ah=0.5, ns=0.2),
        # transforms.RandErasing(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = dataloader(root='./data', train=True, download=False, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    #-------------------#
    #    create model
    #    model type: pre-act-resnet, resnet, wrn
    #-------------------#
    if model_type == "resnet":
        model = resnet(depth=20, num_classes=num_classes)
    elif model_type == "pre-act-resnet":
        # model = preresnet(depth=110, num_classes=num_classes)
        model = preresnet18(num_classes=num_classes)
    else:
        model = wrn(depth=28, num_classes=num_classes, widen_factor=2)  # wrn-28-2

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    # model_train = model.train()

    #-----------------------#
    #    权值加载
    #-----------------------#
    devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_pth != "":
        model.load_state_dict(torch.load(model_pth, map_location=devide))

    #---------------------------------------------#
    #    获取学习率下降公式
    #---------------------------------------------#
    lr_adjust_function = get_lr_scheduler(init_lr, schedule, gamma)

    # -----------------------------#
    #    设置损失函数、优化器
    # -----------------------------#
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    #------------------------------#
    #    创建文件夹
    #------------------------------#
    if not os.path.exists("./logs/cifar-100-" + model_type + "-logs"):
        os.mkdir("./logs/cifar-100-" + model_type + "-logs")
    logfile = "./logs/cifar-100-" + model_type + "-logs"

    best_acc = 0
    logger = Logger(os.path.join(logfile, "cifar-100-resnet.txt"))

    train_epoch_step = len(trainloader)
    test_epoch_step = len(testloader)
    #-----------------------------#
    #    开始训练
    #-----------------------------#
    for epoch in range(Init_epoch, End_epoch):

        for param in model.parameters():
            param.requires_grad = True

        # 获取学习率调整函数
        set_optimizer_lr(optimizer, lr_adjust_function, epoch)

        train_loss, train_acc_top1, train_acc_top5 = train(trainloader, model, criterion, optimizer, epoch, Cuda, train_epoch_step, End_epoch)
        test_loss, test_acc_top1, test_acc_top5 = test(testloader, model, criterion, epoch, Cuda, test_epoch_step, End_epoch)

        # 保存每个世代的训练损失、准确度、学习率
        logger.append([get_lr(optimizer), train_loss, test_loss, train_acc_top1, test_acc_top1, test_acc_top5])

        is_best = test_acc_top1 > best_acc
        best_acc = max(test_acc_top1, best_acc)

        if (epoch + 1) % save_step == 0:
            save_checkpoint(model, is_best, checkpoint=logfile, epoch=epoch + 1)

    logger.closs()


