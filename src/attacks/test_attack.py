import torch

from src.utils.logger import get_logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)


class Test_Attack:
  def __init__(self, attack, testloader, epsilons, eval_steps):
    self.attack = attack
    self.testloader = testloader
    self.device = device
    self.epsilons = epsilons
    self.batch_size = testloader.batch_size
    self.eval_steps = eval_steps

  def test(self):

    accuracies = []
    examples = []  # Run test for each epsilon
    for eps in self.epsilons:
      logger.debug('Performing attack with eps {}'.format(eps))
      acc, ex = self.evaluate(eps, self.eval_steps)
      accuracies.append(acc)
      examples.append(ex)
    return accuracies, examples

  def evaluate(self, epsilon, eval_steps=None):
    total_examples = len(
      self.testloader
    ) * self.batch_size if eval_steps is None else self.batch_size * eval_steps

    eval_step_no = 0
    correct = 0
    total = 0
    adv_examples = []
    n_epochs = 1
    iterator = iter(self.testloader)
    for batch_idx in range(eval_steps):

      if batch_idx == (n_epochs * len(self.testloader)):
        n_epochs = n_epochs + 1
        iterator = iter(self.testloader)
      inputs, targets = iterator.next()
      inputs, targets = inputs.to(device), targets.to(device)

      init_pred, perturbed_data, final_pred = self.attack.generate(
        inputs, epsilon, y=targets)
      total += targets.size(0)
      correct += final_pred.eq(targets).sum().item()
    final_acc = correct / float(total)
    logger.info("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
      epsilon, correct, total, final_acc))

    return final_acc, adv_examples
