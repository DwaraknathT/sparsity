from __future__ import print_function

from abc import ABC, abstractmethod

import torch
from torch.autograd import Variable

from .registry import register

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# logger = get_logger(__name__)

class Attack(ABC):

  def __init__(self, criterion, attack_params):
    self.device = device
    self.min_value = attack_params.min_value
    self.max_value = attack_params.max_value
    self.criterion = criterion

  @abstractmethod
  def generate(self, model, data, epsilon, y=None, y_target=None):
    raise NotImplementedError


@register
class FGSM(Attack):

  def __init__(self, criterion, attack_params):
    """
        parameters:-
        model :-The model under attack
        device :- CPU or GPU acc to usage
        data :- input image
        epsilon :- value of the perturbation
        y :- target /output labels
        targeted :- targeted version of attack

        4 Cases are possible according to the combination of targeted and y variables
        Case 1 :-y is specified and targeted is False .. then y is treated as the real output labels
        Case 2 :-y is specified and targeted is True ... then the targeted version of the attack takes place and y is the target label
        Case 3 :-y is None and targeted is False ... then the predicted outputs of the model are treated as the real outputs and the attack takes place
        Case 4 :-y is None and targeted is True .. Invalid Input"""

    super().__init__(criterion, attack_params)

  def perturb(self,
              model,
              data,
              epsilon,
              output,
              target,
              y_target):
    """
    performs perturbation on the input in accordance with fgsm attack
    inputs :-
          data :- the input image
          epsilon :- the epsilon values with which to perturb
          output :- output values of the network
          target :-target values
          y_target :- None if Untargeted attack


    returns :-the perturbed matrix
    """
    loss = self.criterion(output, target)
    if y_target is not None:
      loss = -loss
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_matrix = epsilon * sign_data_grad
    return perturbed_matrix

  def generate(self,
               model,
               data,
               epsilon,
               y=None,
               y_target=None):
    data = data.to(self.device)
    data.requires_grad = True
    output = model(data, self.hparams)
    init_pred = output.max(1)[1]
    if y_target is not None:  # if no y is specified use predictions as the label for the attack
      target = y_target
    elif y is None:
      target = init_pred
    else:
      target = y  # use y itself as the target
    target = target.to(self.device)
    perturbed_matrix = self.perturb(data, epsilon, output, target, y_target)
    perturbed_data = data + perturbed_matrix
    perturbed_data = torch.clamp(perturbed_data, self.min_value, self.max_value)
    output = model(perturbed_data, self.hparams)
    final_pred = output.max(1)[1]
    return init_pred, perturbed_data, final_pred


@register
class PGD(FGSM):

  def __init__(self, criterion, attack_params):

    self.iters = attack_params.iters
    self.step_size = attack_params.step_size
    self.rand = attack_params.random_start

    super().__init__(criterion, attack_params)

  def generate(self,
               model,
               X_img,
               epsilon,
               y=None,
               y_target=None):
    if self.rand:
      data = X_img + torch.empty(list(X_img.size())).uniform_(
        -epsilon, epsilon).to(self.device)
    else:
      data = X_img.clone()
    data = data.to(self.device)
    output = model(data, self.hparams)
    init_pred = output.max(1)[1]
    if y_target is not None:  # if no y is specified use predictions as the label for the attack
      target = y_target
    elif y is None:
      target = init_pred
    else:
      target = y  # use y itself as the target
    for i in range(self.iters):
      y_var = Variable((target)).to(self.device)
      X_var = data.clone()
      X_var.requires_grad = True
      output = model(X_var, self.hparams)
      perturbed_matrix = self.perturb(X_var, self.step_size, output, y_var,
                                      y_target)
      data = data + perturbed_matrix
      data = torch.clamp(data, self.min_value, self.max_value)
      perturbation = data - X_img
      perturbation = torch.clamp(perturbation, -epsilon, epsilon)
      data = X_img + perturbation
    perturbed_image = data
    perturbed_image = perturbed_image.to(self.device)
    output = model(perturbed_image, self.hparams)
    final_pred = output.max(1)[1]
    return init_pred, perturbed_image, final_pred
