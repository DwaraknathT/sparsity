import copy

import torch
import numpy as np
from src.utils.logger import get_logger
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import seaborn as sns
import matplotlib.pylab as plt

logger = get_logger(__name__)


def heatmap2d(arr, name):
  plt.imshow(arr, cmap='viridis')
  plt.colorbar()
  plt.savefig(name)
  plt.cla()
  plt.clf()


def get_global_masks(grads_abs, unit_grad_norm, reformed_shapes, args):
  # Gather all scores in a single vector and normalise
  if args.prune_type == 'unit':
    all_scores = torch.cat(unit_grad_norm)
    #norm_factor = torch.sum(all_scores)
    #all_scores.div_(norm_factor)
    idx = int(args.final_sparsity * int(all_scores.shape[0]))
    sorted_norms = torch.sort(all_scores)
    acceptable_score = (sorted_norms[0])[idx]
    num_params_to_keep = int(len(all_scores) * (1. - args.final_sparsity))
    masks = []
    for g, reformed_shape, grad in zip(unit_grad_norm, reformed_shapes,
                                       grads_abs):
      w_shape = list(grad.size())
      w = grad.view(w_shape[0], -1).transpose(0, 1)
      norm = torch.norm(w, dim=0)
      #norm.div_(norm_factor)
      mask = (norm < acceptable_score)[None, :]
      mask = mask.repeat(w.shape[0], 1).to(device)
      mask = (1. - mask.float()).transpose(0, 1).view(w_shape)
      masks.append(mask)
  else:
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)
    num_params_to_keep = int(len(all_scores) * (1. - args.final_sparsity))
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]
    masks = []
    for g in grads_abs:
      masks.append(((g / norm_factor) >= acceptable_score).float())

  return masks, len(all_scores), num_params_to_keep


def snip(model, criterion, dataloader, args):
  prune_model = copy.deepcopy(model).to(device)
  masks = []
  addition_masks = []
  for module in prune_model.modules():
    if hasattr(module, 'mask'):
      module.weight.requires_grad = False
      module.mask.requires_grad = True
      masks.append(torch.zeros_like(module.mask.data))
      addition_masks.append(torch.zeros_like(module.mask.data))
    if isinstance(module, torch.nn.BatchNorm2d):
      module.eval()

  if args.union_mask:
    for i, (data, labels) in enumerate(dataloader):
      data, labels = data.to(device), labels.to(device)
      out = prune_model(data)
      loss = criterion(out, labels)
      prune_model.zero_grad()
      loss.backward()

      grads_abs = []
      unit_grad_norm = []
      reformed_shapes = []
      for layer in prune_model.modules():
        if hasattr(layer, 'mask'):
          grad = torch.abs(layer.mask.grad)
          grads_abs.append(grad)
          reshaped_grad = grad.view(grad.shape[0], -1).transpose(0, 1)
          unit_norm = torch.norm(reshaped_grad, dim=0)
          unit_grad_norm.append(unit_norm)
          reformed_shapes.append(reshaped_grad.shape)

          # Reset the params
          layer.reset_parameters()

      point_masks, all_params, num_params_to_keep = get_global_masks(
          grads_abs, unit_grad_norm, reformed_shapes, args)

      mask_sparsities = []
      union_mask_sparsities = []
      for i, point_mask in enumerate(point_masks):
        masks[i] = 1. - ((1. - masks[i]) * (1. - point_mask))
        addition_masks[i] = addition_masks[i] + point_mask
        mask_sparsities.append(
            round(
                1. - np.sum(point_mask.cpu().numpy()) /
                point_mask.cpu().numpy().size, 2))
        union_mask_sparsities.append((round(
            1. - np.sum(masks[i].cpu().numpy()) / masks[i].cpu().numpy().size,
            2)))

    #   logger.info('Mask sparsity {} '.format(mask_sparsities))
    #   logger.info('Union mask sparsity {}'.format(union_mask_sparsities))

  else:
    for i, (data, labels) in enumerate(dataloader):
      if i == args.snip_batch:
        break
      data, labels = data.to(device), labels.to(device)
      out = prune_model(data)
      loss = criterion(out, labels)
      prune_model.zero_grad()
      loss.backward()

    grads_abs = []
    unit_grad_norm = []
    reformed_shapes = []
    for layer in prune_model.modules():
      if hasattr(layer, 'mask'):
        grad = torch.abs(layer.mask.grad)
        grads_abs.append(grad)
        reshaped_grad = grad.view(grad.shape[0], -1).transpose(0, 1)
        unit_norm = torch.norm(reshaped_grad, dim=0)
        unit_grad_norm.append(unit_norm)
        reformed_shapes.append(reshaped_grad.shape)

    masks, all_params, num_params_to_keep = get_global_masks(
        grads_abs, unit_grad_norm, reformed_shapes, args)

  logger.info('Total units in the net {}'.format(all_params))
  logger.info('Units retained after pruning {}'.format(num_params_to_keep))

  del prune_model
  count = 0
  for layer in model.modules():
    if hasattr(layer, 'mask'):
      layer.reset_parameters()
      layer.mask.data = masks[count]
      count += 1
    elif isinstance(layer, torch.nn.Conv2d):
      layer.reset_parameters()
    elif isinstance(layer, torch.nn.Linear):
      layer.reset_parameters()

  return model
