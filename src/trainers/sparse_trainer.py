import torch

from src.attacks.hparams.registry import get_attack_params
from src.attacks.registry import get_attack
from src.attacks.test_attack import Test_Attack
from src.models.registry import register
from src.utils.logger import get_logger
from src.utils.prune import Pruner
from src.utils.utils import get_lr, LrScheduler, save_model, load_model
from src.utils.utils import get_model, mask_check, mask_sparsity
from src.utils.snip import snip

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)

cifar_10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                    'horse', 'ship', 'truck')


@register
class SparseTrainer:

  def __init__(self, args):
    self.args = args
    self.model, self.criterion, self.optimizer = get_model(self.args)
    self.best_acc = 0
    self.step = 0
    if args.resume:
      self.model, self.optimizer, self.step = load_model(
          self.model, self.optimizer, self.args.output_dir, self.args.run_name)

  def test_attack(self, attack, dataloader):
    attack_params = get_attack_params(attack)
    attacker = Test_Attack(attack_params, self.criterion, dataloader)
    attacker.test(self.model)

  def test(self, testloader):
    self.model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total

    self.model.train()
    return loss, acc

  # Training
  def train(self, trainloader, testloader):
    # Prune the model if snip
    if self.args.snip:
      self.model = snip(self.model, self.criterion, trainloader, self.args)
      self.args.end_step = 0

    self.model.train()
    if self.args.steps is None:
      self.args.steps = self.args.epochs * len(trainloader)
    self.args.steps_per_epoch = len(trainloader)
    pruner = Pruner(self.args, self.model)
    scheduler = LrScheduler(self.args, self.optimizer)

    logger.info('Mask check before training')
    mask_check(self.model)

    train_loss = 0
    correct = 0
    total = 0
    n_epochs = 0
    iterator = iter(trainloader)

    for batch_idx in range(self.step, self.args.steps, 1):
      step = batch_idx
      if batch_idx == n_epochs * len(trainloader):
        n_epochs = n_epochs + 1
        iterator = iter(trainloader)

      inputs, targets = iterator.next()
      inputs, targets = inputs.to(device), targets.to(device)
      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = self.criterion(outputs, targets)
      loss.backward()

      self.optimizer.step()
      self.optimizer = scheduler.step(self.optimizer, step)
      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      if batch_idx % self.args.eval_step == 0:
        string = ("Step {} Train Loss: {:.4f} "
                  "Train Accuracy: {:.4f} Learning Rate {:.4f} ".format(
                      step, loss, (correct / total), get_lr(self.optimizer)))
        logger.info(string)
        test_loss, test_acc = self.test(testloader)
        if self.best_acc < test_acc:
          self.best_acc = test_acc
          save_model(step, self.model, self.optimizer, self.args.output_dir,
                     self.args.run_name)
        logger.info("Test Loss: {:.4f} Test Accuracy: {:.4f}".format(
            test_loss, test_acc))
        logger.info('Sparsities {}'.format(mask_sparsity(self.model)))
        logger.info('-------------------')

      self.model = pruner.step(self.model, step)

    logger.info('Training completed')
    test_loss, test_acc = self.test(testloader)
    if self.best_acc < test_acc:
      self.best_acc = test_acc
      save_model(step, self.model, self.optimizer, self.args.output_dir,
                 self.args.run_name)
    logger.info("Final Test Loss: {:.4f} Final Test Accuracy: {:.4f}".format(
        test_loss, test_acc))

    logger.info('Mask check after training')
    mask_check(self.model)
    logger.info("Best test accuracy {:.4f}".format(self.best_acc))

    return self.model
