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
import models
import torch.distributed as dist
from data import DataRegime
from utils.log import setup_logging, ResultsLog, save_checkpoint
from utils.optim import OptimRegime
from utils.cross_entropy import CrossEntropyLoss
from utils.misc import torch_dtypes
from utils.param_filter import FilterModules, is_bn
from datetime import datetime
from ast import literal_eval
from trainer import Trainer

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--datasets-dir', metavar='DATASETS_DIR', default='~/Datasets',
                    help='datasets dir')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input-size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model-config', default='',
                    help='additional architecture configuration')
parser.add_argument('--dtype', default='float',
                    help='type of tensor: ' +
                    ' | '.join(torch_dtypes.keys()) +
                    ' (default: float)')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='rank of distributed processes')
parser.add_argument('--dist-init', default='env://', type=str,
                    help='init used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=-1, type=int,
                    help='mini-batch size (default: same as training)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--label-smoothing', default=0, type=float,
                    help='label smoothing coefficient - default 0')
parser.add_argument('--duplicates', default=1, type=int,
                    help='number of augmentations over singel example')
parser.add_argument('--chunk-batch', default=1, type=int,
                    help='chunk batch size for multiple passes (training)')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='cutout augmentations')
parser.add_argument('--grad-clip', default=-1, type=float,
                    help='maximum grad norm value, -1 for none')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')


def main():
    global args, best_prec1, dtype
    best_prec1 = 0
    args = parser.parse_args()
    dtype = torch_dtypes.get(args.dtype)
    torch.manual_seed(args.seed)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.distributed = args.local_rank >= 0 or args.world_size > 1
    setup_logging(os.path.join(save_path, 'log.txt'),
                  resume=args.resume is not '',
                  dummy=args.distributed and args.local_rank > 0)
    results_path = os.path.join(save_path, 'results')
    results = ResultsLog(
        results_path, title='Training Results - %s' % args.save)

    if args.distributed:
        args.device_ids = [args.local_rank]
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init,
                                world_size=args.world_size, rank=args.local_rank)

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    logging.info("creating model %s", args.model)

    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # create model
    model = models.__dict__[args.model]
    model_config = {'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    # define loss function (criterion) and optimizer
    loss_params = {}
    if args.label_smoothing > 0:
        loss_params['smooth_eps'] = args.label_smoothing
    criterion = getattr(model, 'criterion', CrossEntropyLoss)(**loss_params)
    criterion.to(args.device, dtype)
    model.to(args.device, dtype)

    # Batch-norm should always be done in float
    if 'half' in args.dtype:
        FilterModules(model, module=is_bn).to(dtype=torch.float)

    # optimizer configuration
    optim_regime = getattr(model, 'regime', [{'epoch': 0,
                                              'optimizer': args.optimizer,
                                              'lr': args.lr,
                                              'momentum': args.momentum,
                                              'weight_decay': args.weight_decay}])

    optimizer = optim_regime if isinstance(optim_regime, OptimRegime) \
        else OptimRegime(model, optim_regime, use_float_copy='half' in args.dtype)

    trainer = Trainer(model, criterion, optimizer,
                      device_ids=args.device_ids, device=args.device, dtype=dtype,
                      distributed=args.distributed, local_rank=args.local_rank,
                      grad_clip=args.grad_clip, print_freq=args.print_freq)

    # Evaluation Data loading code
    args.eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    val_data = DataRegime(getattr(model, 'data_eval_regime', None),
                          defaults={'datasets_path': args.datasets_dir, 'name': args.dataset, 'split': 'val', 'augment': False,
                                    'input_size': args.input_size, 'batch_size': args.eval_batch_size, 'shuffle': False,
                                    'num_workers': args.workers, 'pin_memory': True, 'drop_last': False})

    if args.evaluate:
        results = trainer.validate(val_data.get_loader())
        logging.info(results)
        return

    # Training Data loading code
    train_data = DataRegime(getattr(model, 'data_regime', None),
                            defaults={'datasets_path': args.datasets_dir, 'name': args.dataset, 'split': 'train', 'augment': True,
                                      'input_size': args.input_size,  'batch_size': args.batch_size, 'shuffle': True,
                                      'num_workers': args.workers, 'pin_memory': True, 'drop_last': True,
                                      'distributed': args.distributed, 'duplicates': args.duplicates,
                                      'cutout': {'holes': 1, 'length': 16} if args.cutout else None})

    logging.info('optimization regime: %s', optim_regime)
    trainer.training_steps = args.start_epoch * len(train_data)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.epoch = epoch
        train_data.set_epoch(epoch)
        val_data.set_epoch(epoch)
        logging.info('\nStarting Epoch: {0}\n'.format(epoch + 1))

        # train for one epoch
        train_results = trainer.train(train_data.get_loader(),
                                      duplicates=args.duplicates,
                                      chunk_batch=args.chunk_batch)

        # evaluate on validation set
        val_results = trainer.validate(val_data.get_loader())

        if args.distributed and args.local_rank > 0:
            continue

        # remember best prec@1 and save checkpoint
        is_best = val_results['prec1'] > best_prec1
        best_prec1 = max(val_results['prec1'], best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1
        }, is_best, path=save_path)

        logging.info('\nResults - Epoch: {0}\n'
                     'Training Loss {train[loss]:.4f} \t'
                     'Training Prec@1 {train[prec1]:.3f} \t'
                     'Training Prec@5 {train[prec5]:.3f} \t'
                     'Validation Loss {val[loss]:.4f} \t'
                     'Validation Prec@1 {val[prec1]:.3f} \t'
                     'Validation Prec@5 {val[prec5]:.3f} \t\n'
                     .format(epoch + 1, train=train_results, val=val_results))

        values = dict(epoch=epoch + 1, steps=trainer.training_steps)
        values.update({'training ' + k: v for k, v in train_results.items()})
        values.update({'validation ' + k: v for k, v in val_results.items()})
        results.add(**values)

        results.plot(x='epoch', y=['training loss', 'validation loss'],
                     legend=['training', 'validation'],
                     title='Loss', ylabel='loss')
        results.plot(x='epoch', y=['training error1', 'validation error1'],
                     legend=['training', 'validation'],
                     title='Error@1', ylabel='error %')
        results.plot(x='epoch', y=['training error5', 'validation error5'],
                     legend=['training', 'validation'],
                     title='Error@5', ylabel='error %')
        if 'grad' in train_results.keys():
            results.plot(x='epoch', y=['training grad'],
                         legend=['gradient L2 norm'],
                         title='Gradient Norm', ylabel='value')
        results.save()


if __name__ == '__main__':
    main()
