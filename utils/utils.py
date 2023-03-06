import os
import torch
import shutil
from functools import partial

def get_lr_scheduler(lr, schedule, gamma):
    def lr_adjust_function(lr, schedule, gamma, epoch):
        if (epoch + 1) >= schedule[0] and (epoch + 1) < schedule[1]:
            lr *= gamma
        elif (epoch + 1) >= schedule[1]:
            lr *= gamma**2
        return lr
    return partial(lr_adjust_function, lr, schedule, gamma)

def set_optimizer_lr(optimizer, lr_adjust_function, epoch):
    lr = lr_adjust_function(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_szie = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[0: k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_szie))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, path, title=None):
        self.file = None
        self.path = path
        self.title = title
        self.class_name = ["lr", "trl", "tel", "trA-top1", "teA-top1", "teA-top5"]
        if self.path !="":
            self.file = open(self.path, "w")
            for i in self.class_name:
                self.file.write(i+"\t")
            self.file.write("\n")
            self.closs()
            self.file = open(self.path, "a")
    def append(self, log_list):
        for i in log_list:
            self.file.write(str(i) + "\t")
        self.file.write("\n")

    def closs(self):
        self.file.close()

def save_checkpoint(model, is_best, checkpoint=None, epoch=None, filename='model_weights.pth'):
    filepath = os.path.join(checkpoint, filename[:-4] + str(epoch) + filename[-4:])
    torch.save(model.state_dict(), filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))

if __name__ == "__main__":
    logger = Logger("../logs/cifar-100-resnet-20.txt")
    log_list = [0.1, 2.0542, 0.2154, 0.3152, 0.2154]
    logger.append(log_list)
    logger.closs()

    # losses = AverageMeter()
    # losses.update(5, 10)
    # losses.update(12, 20)
    # print(losses.avg())