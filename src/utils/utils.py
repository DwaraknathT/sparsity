import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from src.models.registry import get_model as model_fn
from src.utils.logger import get_logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)


def set_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer


def save_model(step, net, optimizer, directory, filename):
  logger.info('Saving..')
  state = {
      'net': net.state_dict(),
      'optimizer': optimizer.state_dict(),
      'step': step
  }
  if not os.path.isdir(directory):
    os.makedirs(directory)

  torch.save(state, '{}/{}.t7'.format(directory, filename))


def load_model(net, optim, path, name):
  try:
    logger.info('Loading saved model..')
    prev_model = '{}/{}.t7'.format(path, name)
    checkpoint = torch.load(prev_model)
    net.load_state_dict(checkpoint['net'])
    optim.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
  except FileNotFoundError:
    logger.error('Model not found')
    raise FileNotFoundError

  return net, optim, step


class LrScheduler:

  def __init__(self, args, optimizer):
    self.args = args
    self.steps = args.steps
    self.init_lr = args.lr
    if args.lr_schedule == 'cyclic':
      self.cyclic_lr = torch.optim.lr_scheduler.CyclicLR(
          optimizer,
          base_lr=0,
          max_lr=args.lr,
          cycle_momentum=False,
          step_size_up=args.up_step,
          step_size_down=args.down_step)

  def linear_schedule(self, step):
    optim_factor = 0
    if (step > (160 * self.agrs.steps_per_epoch)):
      optim_factor = 3
    elif (step > (120 * self.args.steps_per_epoch)):
      optim_factor = 2
    elif (step > (60 * self.args.steps_per_epoch)):
      optim_factor = 1

    return self.args.lr * math.pow(0.2, optim_factor)

  def step(self, optimizer, step):
    if self.args.lr_schedule == 'linear':
      lr = self.linear_schedule(step)
      set_lr(optimizer, lr)
    elif self.args.lr_schedule == 'cyclic':
      self.cyclic_lr.step()
    else:
      raise NotImplementedError('Only use cyclic, linear, step')
    return optimizer


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def mask_sparsity(net):
  sparsities = []
  for module in net.modules():
    if hasattr(module, 'mask'):
      mask = module.mask.detach().cpu().numpy()
      sparsities.append(round(1. - np.sum(mask) / mask.size, 2))
  return sparsities


def get_model(args):
  model = model_fn(args)
  model = model.to(device)
  params = []
  for n, m in model.named_parameters():
    if 'mask' not in n:
      params.append(m)

  if args.optim == 'sgd':
    optimizer = optim.SGD(params,
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)
  elif args.optim == 'adam':
    optimizer = optim.Adam(params, lr=args.lr)

  criterion = nn.CrossEntropyLoss()
  if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

  return model, criterion, optimizer


# Mask sanity check
def mask_check(net):
  all_ones = False
  zeros_and_ones = False
  mixed_mask = False
  for module in net.modules():
    if hasattr(module, 'mask'):
      mask = module.mask.detach().cpu().numpy()
      # mask = mask.tolist()
      if set(mask.flatten()) == {1}:
        all_ones = True
      elif set(mask.flatten()) == {1, 0}:
        zeros_and_ones = True
      else:
        mixed_mask = True
  if all_ones:
    logger.warning('Mask is all 1s')
  elif zeros_and_ones:
    logger.info('Mask is 0s and 1s')
  elif mixed_mask:
    logger.warning('Mask is not binary')
