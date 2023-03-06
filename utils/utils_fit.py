import torch
from tqdm import tqdm
from .utils import AverageMeter, accuracy, get_lr

def train(trainloader, model, criterion, optimizer, epoch, Cuda, epoch_step, Epoch):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print("start train")
    pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1} / {Epoch}", postfix=dict, mininterval=0.3)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx >= epoch_step:
            break
        if Cuda:
            inputs, targets = inputs.cuda(0), targets.cuda(0)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # print(outputs.size())
        # print(targets.size())

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(**{
            'loss': losses.avg,
            'acc': top1.avg,
            'lr': get_lr(optimizer)
        })
        pbar.update(1)
    pbar.close()
    print("finish train")
    return [losses.avg, top1.avg, top5.avg]

def test(testloader, model, criterion, epoch, Cuda, epoch_step, Epoch):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print("start test")
    pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1} / {Epoch}", postfix=dict, mininterval=0.3)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx >= epoch_step:
            break
        if Cuda:
            inputs, targets = inputs.cuda(0), targets.cuda(0)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        pbar.set_postfix(**{
            'loss': losses.avg,
            'acc': top1.avg
        })
        pbar.update(1)
    pbar.close()
    print("finish test")
    return (losses.avg, top1.avg, top5.avg)