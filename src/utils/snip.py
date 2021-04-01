import copy

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def snip(model, criterion, dataloader, args):
    prune_model = copy.deepcopy(model).to(device)
    for module in prune_model.modules():
        if hasattr(module, "mask"):
            module.weight.requires_grad = False
            module.mask.requires_grad = True
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
    for layer in prune_model.modules():
        if hasattr(layer, "mask"):
            grad = torch.abs(layer.mask.grad)
            grads_abs.append(grad)
            reshaped_grad = grad.view(grad.shape[0], -1).transpose(0, 1)
            unit_norm = torch.norm(reshaped_grad, dim=0)
            unit_grad_norm.append(unit_norm)
            reformed_shapes.append(reshaped_grad.shape)

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
        if hasattr(layer, "mask"):
            layer.reset_parameters()
            layer.mask.data = masks[count]
            count += 1
        elif isinstance(layer, torch.nn.Conv2d):
            layer.reset_parameters()
        elif isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()

    return model
