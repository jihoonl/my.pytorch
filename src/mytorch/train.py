import time

import torch
import torch.nn.functional as F

from .utils.logger import logger
from .utils.config import cfg
from .data import loader
from .model import create_model, get_trainable_params
from .optimizer import get_optimizer
from .utils.timer import Timer
from .utils.fileio import export


def train_model(data):
    train_start_time = time.strftime("%b%d-%H%M")

    device = cfg.DEVICE
    epoch = cfg.TRAIN.EPOCH
    multi_gpu = cfg.GPU.MULTI
    train_loader = loader.train(data['train'])
    test_loader = loader.test(data['test'])
    model = create_model(model_config=cfg.TRAIN.MODEL, multi_gpu=multi_gpu)

    trainable_params = get_trainable_params(model, multi_gpu)
    opt = get_optimizer(trainable_params)
    model.to(device)

    t = Timer()
    l = None
    for e in range(1, epoch + 1):
        t.tic()
        train(model, train_loader, device, opt)
        lr = opt.param_groups[0]['lr']
        elapse = t.toc()
        train_loss, train_accuracy = test(model, train_loader, device)
        test_loss, test_accuracy = test(model, test_loader, device)
        l = '{:>10.4f}, {:>10.6f}, {:>10.4f}, {:>10.6f}'.format(test_accuracy, test_loss, train_accuracy, train_loss)
        log = ('[{:2}/{:2}][{:.3f}s][{:.6f}] - '
               '(Train, Test) '
               'Loss: {:.6f} - {:.6f}, '
               'Acc: {:.4f} - {:.4f}').format(e, epoch, elapse, lr, train_loss,
                                              test_loss, train_accuracy,
                                              test_accuracy)
        logger.info(log)
    export(model, l, multi_gpu,  train_start_time)


def train(model, dataloader, device, optimizer):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    return test_loss, accuracy
