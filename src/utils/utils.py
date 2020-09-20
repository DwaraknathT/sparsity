import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from src.models.registry import get_model as model_fn
from src.utils.args import get_args
from src.utils.logger import get_logger

args = get_args()
logger = get_logger(__name__)


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def set_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer


def get_model():
  model = model_fn(args)
  model = model.to(args.device)
  params = []
  for n, m in model.named_parameters():
    if 'mask' not in n:
      params.append(m)

  if args.optim == 'sgd':
    optimizer = optim.SGD(
      params,
      lr=args.lr,
      momentum=0.9,
      weight_decay=5e-4,
      nesterov=True
    )
  elif args.optim == 'adam':
    optimizer = optim.Adam(
      params,
      lr=args.lr)

  if args.lr_schedule == 'step':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer,
      args.milestones
    )
  elif args.lr_schedule == 'cyclic':
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
      optimizer,
      base_lr=0,
      max_lr=args.lr,
      step_size_up=args.up_step,
      step_size_down=args.down_step)

  criterion = nn.CrossEntropyLoss()
  if args.device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

  return model, criterion, optimizer, lr_scheduler


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
