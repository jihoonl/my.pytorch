import time

import torch
import torch.nn.functional as F
from PIL import Image

from .data import get_dataset, get_preprocessed_data, loader
from .model import get_trainable_params, load_model
from .optimizer import get_lr_scheduler, get_optimizer
from .utils.config import cfg
from .utils.fileio import export
from .utils.logger import Progressbar, logger
from .utils.tensorboard import get_writer
from .utils.timer import Timer


def train_model():
    train_start_time = time.strftime("%b%d-%H%M")

    device = cfg.DEVICE
    epoch = cfg.TRAIN.EPOCH
    multi_gpu = cfg.GPU.MULTI
    resume = cfg.RESUME if cfg.RESUME.CHECKPOINT else None

    data = get_dataset(cfg.DATASET)

    train_loader = loader.train(data['train'])
    test_loader = loader.test(data['test'])
    model = load_model(model_config=cfg.MODEL,
                       resume_config=resume,
                       multi_gpu=multi_gpu)

    trainable_params = get_trainable_params(model, multi_gpu)
    opt = get_optimizer(trainable_params, cfg.TRAIN.OPTIMIZER)
    lr_scheduler = get_lr_scheduler(opt, cfg.TRAIN.LR_SCHEDULER)
    writer = get_writer(train_start_time, cfg)

    model.to(device)

    t = Timer()
    l = None
    for e in range(1, epoch + 1):
        t.tic()
        train(model, train_loader, device, opt, e, epoch, writer=writer)
        lr = opt.param_groups[0]['lr']
        elapse = t.toc()

        if e % cfg.TRAIN.EVAL_EPOCH == 0:
            if cfg.TRAIN.EVAL_TRAIN:
                train_loss, train_accuracy = evaluate(model, train_loader,
                                                      device, e, epoch,
                                                      'Eval[T]', writer)
                eval_loss, eval_accuracy = evaluate(model, test_loader, device,
                                                    e, epoch, 'Eval[V]', writer)
                l = '{:>10.4f}, {:>10.6f}, {:>10.4f}, {:>10.6f}'.format(
                    eval_accuracy, eval_loss, train_accuracy, train_loss)
            else:
                eval_loss, eval_accuracy = evaluate(model, test_loader, device,
                                                    e, epoch, 'Eval[V]', writer)
                l = '{:>10.4f}, {:>10.6f}'.format(eval_accuracy, eval_loss)
            """
            log = ('[{:2}/{:2}][{:.3f}s][{:.6f}] - '
                   '(Train, Test) '
                   'Loss: {:.6f} - {:.6f}, '
                   'Acc: {:.4f} - {:.4f}').format(e, epoch, elapse, lr,
                                                  train_loss, test_loss,
                                                  train_accuracy, test_accuracy)
            logger.info(log)
            """
    export(model, l, multi_gpu, train_start_time)


def train(model,
          dataloader,
          device,
          optimizer,
          epoch,
          max_epoch,
          name='Train',
          writer=None):

    model.train()
    loss_sum = 0
    loss = 0
    loader = Progressbar(dataloader, name, epoch, max_epoch)
    for data, target in loader():
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss
        desc = {
            'Loss ': ' {:5.4f}'.format(loss),
        }
        loader.desc(desc)
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Train/loss', loss_sum / len(dataloader.dataset), epoch)
    writer.add_scalar('Train/lr', lr, epoch)


def evaluate(model,
             dataloader,
             device,
             epoch,
             max_epoch,
             name='Eval',
             writer=None):
    model.eval()
    eval_loss = 0
    correct = 0

    with torch.no_grad():
        loader = Progressbar(dataloader, name, epoch, max_epoch)
        len_data_cum = 0
        for data, target in loader():
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            eval_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            len_data_cum += len(data)
            desc = {
                'Loss ': ' {:5.4f}'.format(eval_loss / len_data_cum),
                'Acc  ': ' {:3.4f}'.format(correct / len_data_cum)
            }
            loader.desc(desc)
    eval_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    writer.add_scalar('Eval/loss', eval_loss)
    writer.add_scalar('Eval/acc', accuracy)

    return eval_loss, accuracy


def inference_from_file(img_file):
    img = Image.open(img_file)

    device = cfg.DEVICE
    multi_gpu = cfg.GPU.MULTI
    resume = cfg.RESUME if cfg.RESUME.CHECKPOINT else None

    if not resume:
        raise Exception('Cannot inference if resume is None, {}'.format(
            cfg.RESUME))

    data = get_preprocessed_data(img, cfg.DATASET)
    model = load_model(model_config=cfg.MODEL,
                       resume_config=resume,
                       multi_gpu=multi_gpu)
    model.eval()

    with torch.no_grad():
        model.to(device)
        data.to(device)
        output = model(torch.unsqueeze(data, 0))
        output = F.log_softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
    return int(pred[0])
