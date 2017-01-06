import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from importlib import import_module
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    help='model architecture: alexnet | resnet | ... '
                         '(default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    log = logging.getLogger('main')
    log.info("saving to %s", save_path)
    log.info("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])

    # create model
    log.info("creating model %s", args.model)
    model = import_module('.' + args.model, 'models').model
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    log.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        log.info("loaded checkpoint '%s' (epoch %s)",
                 args.evaluate, checkpoint['epoch'])
    elif args.resume:
        if os.path.isfile(args.resume):
            log.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            log.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
        else:
            log.error("no checkpoint found at '%s'", args.resume)

    cudnn.benchmark = True
    num_parameters = sum([l.nelement() for l in model.parameters()])
    log.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().type(args.type)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    log.info('training regime: %s', regime)

    results = ResultsLog(fields_list=['epoch', 'train_loss', 'val_loss',
                                      'train_prec1', 'val_prec1', 'train_prec5', 'val_prec5'],
                         format_list=['d'] + 6 * ['.3f'],
                         path=os.path.join(save_path, 'results.csv'))

    if 'cuda' in args.type and len(args.gpus)>1:
        model = torch.nn.DataParallel(model, args.gpus)

    model.type(args.type)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer)

        kernel_img = model.features[0][0].kernel.data.clone()
        kernel_img.add_(-kernel_img.min())
        kernel_img.mul_(255/kernel_img.max())
        save_image(kernel_img, 'kernel%s.jpg' % epoch)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        log.info('\n Epoch: {0}\t'
                 'Training Loss {train_loss:.4f} \t'
                 'Training Prec@1 {train_prec1:.3f} \t'
                 'Training Prec@5 {train_prec5:.3f} \t'
                 'Validation Loss {val_loss:.4f} \t'
                 'Validation Prec@1 {val_prec1:.3f} \t'
                 'Validation Prec@5 {val_prec5:.3f} \n'
                 .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                         train_prec1=train_prec1, val_prec1=val_prec1,
                         train_prec5=train_prec5, val_prec5=val_prec5))

        results.info(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                     train_prec1=train_prec1, val_prec1=val_prec1,
                     train_prec5=train_prec5, val_prec5=val_prec5)
        results.plot(os.path.join(save_path, 'results.html'))


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    log = logging.getLogger('main')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = Variable(input, volatile=not training).type(args.type)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     log.info('TRAIN', extra={'epoch':epoch, 'current':i, 'total':len(data_loader),
        #                        'time':batch_time.avg, 'data_time':data_time.avg,
        #                        'loss':losses.avg, 'top1':top1.avg, 'top5':top5.avg})
        if i % args.print_freq == 0:
            log.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         epoch, i, len(data_loader), phase='TRAINING' if training else 'EVALUATING', batch_time=batch_time,
                         data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
