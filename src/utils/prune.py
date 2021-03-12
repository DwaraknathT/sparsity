import math

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prune(w, perc, prune_type):
  if prune_type == 'weight':
    w_shape = list(w.size())
    w = w.view(w_shape[0], -1).transpose(0, 1)
    norm = torch.abs(w)
    idx = int(perc * w.shape[0])
    threshold = (torch.sort(norm, dim=0)[0])[idx]
    mask = (norm < threshold).to(device)
    mask = (1. - mask.float()).transpose(0, 1).view(w_shape)
  elif prune_type == 'unit':
    w_shape = list(w.size())
    w = w.view(w_shape[0], -1).transpose(0, 1)
    norm = torch.norm(w, dim=0)
    idx = int(perc * int(w.shape[1]))
    sorted_norms = torch.sort(norm)
    threshold = (sorted_norms[0])[idx]
    mask = (norm < threshold)[None, :]
    mask = mask.repeat(w.shape[0], 1).to(device)
    mask = (1. - mask.float()).transpose(0, 1).view(w_shape)
  else:
    raise NotImplementedError('Pruning type not implemented.')

  return mask


class Pruner:

  def __init__(self, args, model):
    self.args = args
    self.initial_masks = []
    self.union_masks = []
    self.total_masks = 0
    for module in model.modules():
      if hasattr(module, 'mask'):
        self.initial_masks.append(module.mask.data)
        self.union_masks.append(torch.zeros_like(module.mask.data))

    self.start_step = int(args.start_step * args.steps)
    self.end_step = int(args.end_step * args.steps)
    if self.args.ramp_cycle_step is None:
      self.args.ramp_cycle_step = args.down_step

  def compute_sparsity(self, step):
    if self.args.ramp_type == "linear":
      rate_of_increase = (self.args.final_sparsity - self.args.initial_sparsity
                         ) / (self.end_step - self.start_step)
      prune_compute = self.args.initial_sparsity + rate_of_increase * (
          step - self.start_step)
    elif self.args.ramp_type == "constant":
      prune_compute = self.args.final_sparsity
    elif self.args.ramp_type == "half_cycle":
      sin_inner = (math.pi / 2 * (step % self.args.ramp_cycle_step) /
                   self.args.ramp_cycle_step)
      prune_compute = (self.args.final_sparsity * math.sin(sin_inner))
    elif self.args.ramp_type == "full_cycle":
      sin_inner = (math.pi / 2 * (step / self.args.ramp_cycle_step))
      prune_compute = (self.args.final_sparsity * abs(math.sin(sin_inner)))

    return prune_compute

  def global_prune(self, model, prune_compute):
    weights = []
    for layer in model.modules():
      if hasattr(layer, 'mask'):
        weights.append(layer.weight)
    scores = torch.cat([torch.flatten(w) for w in weights])
    idx = int(prune_compute * scores.shape[0])
    norm = torch.abs(scores)
    threshold = (torch.sort(norm, dim=0)[0])[idx]
    # threshold = values[-1]
    masks = [(torch.abs(w) > threshold).float() for w in weights]
    count = 0
    for layer in model.modules():
      if hasattr(layer, 'mask'):
        layer.mask.data = masks[count]
        count += 1
    return model

  def local_prune(self, model, prune_compute):
    for module in model.modules():
      if hasattr(module, 'mask'):
        mask_sparsity = round(
            1. - np.sum(module.mask.detach().cpu().numpy()) /
            module.mask.detach().cpu().numpy().size, 2)
        if mask_sparsity <= self.args.final_sparsity and mask_sparsity <= prune_compute:
          if self.args.carry_mask:
            module_mask = prune(module.weight * module.mask, prune_compute,
                                self.args.prune_type)
          else:
            module_mask = prune(module.weight, prune_compute,
                                self.args.prune_type)
          module.mask.data = module_mask
    return model

  def union_mask(self, model, step):
    # save masks
    sparsities = []
    count = 0
    if (step < (self.end_step - self.args.ramp_cycle_step)):
      if (step % self.args.ramp_cycle_step == 0) and (step != 0):
        logger.info('Saving mask')
        for module in model.modules():
          if hasattr(module, 'mask'):
            mask_sparsity = round(
                1. - np.sum(module.mask.detach().cpu().numpy()) /
                module.mask.detach().cpu().numpy().size, 2)
            self.union_masks[count] = 1. - ((1. - self.union_masks[count]) *
                                            (1. - module.mask.data))
            union_mask_sparsity = (round(
                1. - np.sum(self.union_masks[count].detach().cpu().numpy()) /
                self.union_masks[count].detach().cpu().numpy().size, 2))
            # Reset the masks
            module.mask.data = torch.ones_like(module.mask.data)
            self.total_masks += 1
            count += 1
            logger.info(
                'Mask sparsity {:.2f} Union mask sparsity {:.2f}'.format(
                    mask_sparsity, union_mask_sparsity))
    elif (step == (self.end_step - self.args.ramp_cycle_step)):
      logger.info('Updating masks to union mask')
      for module in model.modules():
        if hasattr(module, 'mask'):
          module.mask.data = self.union_masks[count]
          mask_sparsity = round(
              1. - np.sum(module.mask.detach().cpu().numpy()) /
              module.mask.detach().cpu().numpy().size, 2)
          sparsities.append(mask_sparsity)
          count += 1
      self.args.initial_sparsity = min(sparsities)
      logger.info('Initial sparsity updated to {}'.format(
          self.args.initial_sparsity))
      logger.info('Setting carry mask to true')
      self.args.carry_mask = True
      self.args.ramp_type = 'linear'
      self.start_step = (self.end_step - self.args.ramp_cycle_step)

    return model

  def ramping_prune(self, model, step):
    prune_percent = self.compute_sparsity(step)
    if self.args.union_mask:
      model = self.union_mask(model, step)
    if step == self.end_step:
      prune_percent = self.args.final_sparsity
    if self.start_step <= step < self.end_step:
      if step % self.args.prune_freq == 0:
        if self.args.global_prune:
          model = self.global_prune(model, prune_percent)
        else:
          model = self.local_prune(model, prune_percent)

    return model

  def single_shot_prune(self, model, step):
    if step == self.start_step:
      if self.args.global_prune:
        model = self.global_prune(model, self.args.final_sparsity)
      else:
        model = self.local_prune(model, self.args.final_sparsity)

    return model

  def step(self, model, step):
    if self.args.ramping:
      model = self.ramping_prune(model, step)
    else:
      model = self.single_shot_prune(model, step)

    return model
