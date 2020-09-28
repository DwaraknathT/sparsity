import torch

from src.attacks.hparams.registry import get_attack_params
from src.attacks.registry import get_attack
from src.attacks.test_attack import Test_Attack
from src.models.registry import register
from src.utils.logger import get_logger
from src.utils.utils import LrScheduler, save_model, load_model
from src.utils.utils import get_lr, get_model, mask_check

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)

cifar_10_classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


@register
class DenseTrainer:
  def __init__(self, args):
    self.args = args
    self.model, self.criterion, self.optimizer = get_model(self.args)
    self.best_acc = 0
    if args.load_model:
      self.model = load_model(self.model, self.args.output_dir, self.args.run_name)

  def test_attack(self, attack, dataloader):
    attack_params = get_attack_params(attack)
    attack = get_attack(self.criterion, attack_params)
    attacker = Test_Attack(attack,
                           dataloader,
                           attack_params.epsilons,
                           attack_params.eval_steps)
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
    # Get self.model, self.optimizer, self.criterion, lr_scheduler

    self.model.train()
    if self.args.steps is None:
      self.args.steps = self.args.epochs * len(trainloader)

    logger.info('Mask check before training')
    mask_check(self.model)
    scheduler = LrScheduler(self.args)

    train_loss = 0
    correct = 0
    total = 0
    n_epochs = 0
    iterator = iter(trainloader)

    for batch_idx in range(0, self.args.steps, 1):
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
                  "Train Accuracy: {:.4f} Learning Rate {:.4f} ".format(step,
                                                                        loss,
                                                                        (correct / total),
                                                                        get_lr(self.optimizer)))
        logger.info(string)
        test_loss, test_acc = self.test(testloader)
        if self.best_acc < test_acc:
          self.best_acc = test_acc
          save_model(self.model, self.optimizer, self.args.output_dir, self.args.run_name)
        logger.info("Test Loss: {:.4f} Test Accuracy: {:.4f}".format(test_loss,
                                                                     test_acc))

    logger.info('Training completed')
    test_loss, test_acc = self.test(testloader)
    if self.best_acc < test_acc:
      self.best_acc = test_acc
      save_model(self.model, self.optimizer, self.args.output_dir, self.args.run_name)
    logger.info("Final Test Loss: {:.4f} Final Test Accuracy: {:.4f}".format(test_loss,
                                                                             test_acc))
    logger.info("Best test accuracy {:.4f}".format(self.best_acc))

    return self.model
