import torch
import torch.nn.functional as F

from mytorch.utils.logger import logger
from mytorch.utils.config import cfg
from mytorch.data import loader
from mytorch.model.mnist_base import MnistBaseNet
from mytorch.optimizer import get_optimizer
from mytorch.utils.timer import Timer


def train_model(data):

    device = cfg.DEVICE
    epoch = cfg.TRAIN.EPOCH
    train_loader = loader.train(data['train'])
    test_loader = loader.test(data['test'])
    model = MnistBaseNet((28, 28))
    opt = get_optimizer(model.parameters())

    if cfg.GPU.MULTI:
        model = torch.nn.DataParallel(model)
    model.to(device)
    logger.info(opt)

    t = Timer()
    for e in range(1, epoch + 1):
        t.tic()
        train(model, train_loader, device, opt)
        lr = opt.param_groups[0]['lr']
        elapse = t.toc()
        train_loss, train_accuracy = test(model, train_loader, device)
        test_loss, test_accuracy = test(model, test_loader, device)
        logger.info('[{:2}/{:2}][{:.3f}s][{:.6f}] - '
                    '(Train, Test) '
                    'Loss: {:.6f} - {:.6f}, '
                    'Acc: {:.4f} - {:.4f}'.format(
                        e, epoch, elapse, lr, train_loss, test_loss,
                        train_accuracy, test_accuracy))


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
