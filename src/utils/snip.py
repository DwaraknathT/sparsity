import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import MaskedConv, MaskedDense

device = "cuda" if torch.cuda.is_available() else "cpu"

import types


def snip_forward_conv2d(self, x):
    input_shape = x.shape
    self.compute = (
        2 * input_shape[2] * input_shape[3] * input_shape[1] * self.kernel_size[0] ** 2
    )
    return F.conv2d(
        x,
        self.weight * self.mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self, x):
    input_shape = x.shape
    self.compute = (
        2 * input_shape[2] * input_shape[3] * input_shape[1] * torch.pow(self.in_dim, 2)
    )
    return F.linear(x, self.weight * self.mask, self.bias)


def snip(model, criterion, dataloader, args):
    prune_model = copy.deepcopy(model).to(device)
    for module in prune_model.modules():
        if hasattr(module, "mask"):
            module.weight.requires_grad = False
            module.mask.requires_grad = True
            module.compute = 0
            if isinstance(module, MaskedConv):
                module.forward = types.MethodType(snip_forward_conv2d, module)
            if isinstance(module, MaskedDense):
                module.forward = types.MethodType(snip_forward_linear, module)

        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

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
    computes = []
    for layer in prune_model.modules():
        if hasattr(layer, "mask"):
            grad = torch.abs(layer.mask.grad)
            grads_abs.append(grad)
            computes.append(layer.compute)
            reshaped_grad = grad.view(grad.shape[0], -1).transpose(0, 1)
            unit_norm = torch.norm(reshaped_grad, dim=0)
            unit_grad_norm.append(unit_norm)
            reformed_shapes.append(reshaped_grad.shape)

    if args.compute_aware:
        compute_aware_grad_abs = []
        total_compute = sum(computes)
        for grad, compute in zip(grads_abs, computes):
            compute_hat = compute / total_compute
            normalized_compute = compute_hat / max(computes)

            if args.compute_aware:
                compute_aware_grad = grad / normalized_compute
                compute_aware_grad_abs.append(compute_aware_grad)
                reshaped_grad = compute_aware_grad.view(
                    compute_aware_grad.shape[0], -1
                ).transpose(0, 1)
            else:
                reshaped_grad = grad.view(grad.shape[0], -1).transpose(0, 1)
            unit_norm = torch.norm(reshaped_grad, dim=0)
            unit_grad_norm.append(unit_norm)
            reformed_shapes.append(reshaped_grad.shape)
        grads_abs = compute_aware_grad_abs

    # Gather all scores in a single vector and normalise
    if args.prune_type == "unit":
        all_scores = torch.cat(unit_grad_norm)
        # norm_factor = torch.sum(all_scores)
        # all_scores.div_(norm_factor)
        idx = int(args.final_sparsity * int(all_scores.shape[0]))
        sorted_norms = torch.sort(all_scores)
        acceptable_score = (sorted_norms[0])[idx]
        num_params_to_keep = int(len(all_scores) * (1.0 - args.final_sparsity))
        print("Total units in the net {}".format(len(all_scores)))
        print("Units retained after pruning {}".format(num_params_to_keep))
        masks = []
        for g, reformed_shape, grad in zip(unit_grad_norm, reformed_shapes, grads_abs):
            w_shape = list(grad.size())
            w = grad.view(w_shape[0], -1).transpose(0, 1)
            norm = torch.norm(w, dim=0)
            # norm.div_(norm_factor)
            mask = (norm < acceptable_score)[None, :]
            mask = mask.repeat(w.shape[0], 1).to(device)
            mask = (1.0 - mask.float()).transpose(0, 1).view(w_shape)
            masks.append(mask)
    else:
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
        num_params_to_keep = int(len(all_scores) * (1.0 - args.final_sparsity))
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        print("Total units in the net {}".format(len(all_scores)))
        print("Units retained after pruning {}".format(num_params_to_keep))
        masks = []
        for g in grads_abs:
            masks.append(((g / norm_factor) >= acceptable_score).float())

    del prune_model
    count = 0
    for layer in model.modules():
        # print(masks[count].shape)
        # mask_array = masks[count].cpu().numpy()
        # w = masks[count].view(masks[count].shape[0], -1).transpose(0, 1)
        # w = torch.norm(w, dim=0)
        # indices = torch.nonzero(w, as_tuple=True)[0].cpu().numpy()
        # reduced_array = np.take(mask_array, indices, axis=0)
        # print(reduced_array.shape)
        # print("--------")
        if hasattr(layer, "mask"):
            layer.reset_parameters()
            layer.mask.data = masks[count]
            count += 1
        elif isinstance(layer, torch.nn.Conv2d):
            layer.reset_parameters()
        elif isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()

    return model
