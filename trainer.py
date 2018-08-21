import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
from utils.meters import AverageMeter, accuracy


class Trainer(object):

    def __init__(self, model, criterion, optimizer=None,
                 device_ids=0, device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1,
                 grad_clip=-1, print_freq=100):
        self._model = model
        self.criterion = criterion
        self.epoch = 0
        self.training_steps = 0
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.grad_clip = grad_clip

        def empty_reg(m): return 0
        self.regularizer = getattr(model, 'regularization', empty_reg)
        self.regularizer_pre_step = getattr(
            model, 'regularization_pre_step', empty_reg)
        self.regularizer_post_step = getattr(
            model, 'regularization_post_step', empty_reg)
        if distributed:
            self.model = nn.parallel.DistributedDataParallel(model,
                                                             device_ids=[
                                                                 local_rank],
                                                             output_device=local_rank)
        elif device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids)
        else:
            self.model = model

    def _step(self, inputs, target, training=False):
        # compute output
        output = self.model(inputs)
        loss = self.criterion(output, target)
        loss += self.regularizer(self.model)
        grad = None

        if isinstance(output, list) or isinstance(output, tuple):
            output = output[0]

        if training:
            self.optimizer.update(self.epoch, self.training_steps)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.regularizer_pre_step(self.model)
            if self.grad_clip > 0:
                grad = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.regularizer_post_step(self.model)
            self.training_steps += 1

        return output, loss, grad

    def forward(self, data_loader, num_steps=None, training=False):
        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()

        for i, (inputs, target) in enumerate(data_loader):
            # measure data loading time
            meters['data'].update(time.time() - end)
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            output, loss, grad = self._step(inputs, target, training=training)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})'\
                        .format(meters=meters)
                logging.info(report)

            if num_steps is not None and i >= num_steps:
                break

        return meter_results(meters)

    def train(self, data_loader):
        # switch to train mode
        self.model.train()

        return self.forward(data_loader, training=True)

    def validate(self, data_loader):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, training=False)
