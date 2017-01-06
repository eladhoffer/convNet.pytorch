import os
import torch
import json
import logging.config
import shutil
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column


def setup_logging(log_file='log.txt', path='logging.json', level=logging.INFO):
    """Setup logging configuration
    """
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        config['handlers']['log_file']['filename'] = log_file
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=level)


class ResultsLog(object):

    def __init__(self, fields_list,
                 format_list=None, delim=',',
                 path='results.csv',
                 level=logging.INFO):
        self.path = path
        self.fields = fields_list
        format_list = format_list or ['%f'] * len(self.fields)
        self.format = delim.join(
            ['{' + i + ':' + j + '}' for i, j in zip(self.fields, format_list)])
        self.logger = logging.getLogger('results')
        handler = logging.FileHandler(path, mode='w')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.info(delim.join(self.fields))

    def info(self, *kargs, **kwargs):
        self.logger.info(self.format.format(*kargs, **kwargs))

    def plot(self, plot_file=None):
        plot_file = plot_file or (self.path + '.html')
        if os.path.isfile(plot_file):
            os.remove(plot_file)
        PLOT = pd.read_csv(self.path)

        width = 800
        height = 400

        output_file(plot_file, title="Training Results")

        def draw_train_val(epoch, train, val, title):
            g = figure(width=width, height=height, title=title)
            g.line(epoch, train, color='red', legend="training")
            g.circle(epoch, train, color='red', size=8)
            g.line(epoch, val, color='blue', legend="validation")
            g.circle(epoch, val, color='blue', size=8)
            return g

        g1 = draw_train_val(PLOT['epoch'], PLOT['train_loss'],
                            PLOT['val_loss'], 'Loss')
        g2 = draw_train_val(PLOT['epoch'], 100 - PLOT['train_prec1'],
                            100 - PLOT['val_prec1'], 'Error @1')
        g3 = draw_train_val(PLOT['epoch'], 100 - PLOT['train_prec5'],
                            100 - PLOT['val_prec5'], 'Error @5')

        p = column(g1, g2, g3)
        save(p)


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch%s.pth.tar' % state['epoch']))


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

optimizers = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'Adagrad': torch.optim.Adagrad,
    'RMSprop': torch.optim.RMSprop
}


def adjust_optimizer(optimizer, epoch, config, verbose=True):
    """Reconfigures the optimizer according to epoch and config dict"""
    adjustment = None
    if epoch not in config:
        return optimizer
    setting = config[epoch]
    # if 'optimizer' in setting:
    # optimizer = setting['optimizer'](optimizer.params_dict)
    for param_group in optimizer.state_dict()['param_groups']:
        for key in param_group.iterkeys():
            if key in setting:
                param_group[key] = setting[key]
    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
