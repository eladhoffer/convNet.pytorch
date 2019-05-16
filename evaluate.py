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

parser = argparse.ArgumentParser(description='PyTorch ConvNet Evaluation')
parser.add_argument('evaluate', type=str,
                    help='evaluate model FILE on validation set')
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
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--label-smoothing', default=0, type=float,
                    help='label smoothing coefficient - default 0')
parser.add_argument('--mixup', default=None, type=float,
                    help='mixup alpha coefficient - default None')
parser.add_argument('--duplicates', default=1, type=int,
                    help='number of augmentations over singel example')
parser.add_argument('--chunk-batch', default=1, type=int,
                    help='chunk batch size for multiple passes (training)')
parser.add_argument('--augment', action='store_true', default=False,
                    help='perform augmentations')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='cutout augmentations')
parser.add_argument('--autoaugment', action='store_true', default=False,
                    help='use autoaugment policies')
parser.add_argument('--avg-out', action='store_true', default=False,
                    help='average outputs')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    global best_prec1, dtype
    best_prec1 = 0
    dtype = torch_dtypes.get(args.dtype)
    torch.manual_seed(args.seed)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)

    args.distributed = args.local_rank >= 0 or args.world_size > 1

    if not os.path.exists(save_path) and not (args.distributed and args.local_rank > 0):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'),
                  resume=args.resume is not '',
                  dummy=args.distributed and args.local_rank > 0)

    results_path = os.path.join(save_path, 'results')
    results = ResultsLog(
        results_path, title='Training Results - %s' % args.save)

    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    if not os.path.isfile(args.evaluate):
        parser.error('invalid checkpoint: {}'.format(args.evaluate))
    checkpoint = torch.load(args.evaluate, map_location="cpu")
    # Overrride configuration with checkpoint info
    args.model = checkpoint.get('model', args.model)
    args.model_config = checkpoint.get('config', args.model_config)

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    logging.info("creating model %s", args.model)

    # create model
    model = models.__dict__[args.model]
    model_config = {'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # load checkpoint
    model.load_state_dict(checkpoint['state_dict'])
    logging.info("loaded checkpoint '%s' (epoch %s)",
                 args.evaluate, checkpoint['epoch'])

    # define loss function (criterion) and optimizer
    loss_params = {}
    if args.label_smoothing > 0:
        loss_params['smooth_eps'] = args.label_smoothing
    criterion = getattr(model, 'criterion', nn.NLLLoss)(**loss_params)
    criterion.to(args.device, dtype)
    model.to(args.device, dtype)

    # Batch-norm should always be done in float
    if 'half' in args.dtype:
        FilterModules(model, module=is_bn).to(dtype=torch.float)

    trainer = Trainer(model, criterion,
                      device_ids=args.device_ids, device=args.device, dtype=dtype,
                      mixup=args.mixup, print_freq=args.print_freq)

    # Evaluation Data loading code
    val_data = DataRegime(getattr(model, 'data_eval_regime', None),
                          defaults={'datasets_path': args.datasets_dir, 'name': args.dataset, 'split': 'val', 'augment': args.augment,
                                    'input_size': args.input_size, 'batch_size': args.batch_size, 'shuffle': False, 'duplicates': args.duplicates, 'autoaugment': args.autoaugment,
                                    'cutout': {'holes': 1, 'length': 16} if args.cutout else None, 'num_workers': args.workers, 'pin_memory': True, 'drop_last': False})

    results = trainer.validate(val_data.get_loader(),
                               duplicates=val_data.get('duplicates'),
                               average_output=args.avg_out)
    logging.info(results)
    return results


if __name__ == '__main__':
    main()
