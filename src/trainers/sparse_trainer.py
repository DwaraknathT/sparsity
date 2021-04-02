import time

import torch

from src.attacks.hparams.registry import get_attack_params
from src.attacks.registry import get_attack
from src.attacks.test_attack import Test_Attack
from src.models.registry import register
from src.utils.logger import get_logger
from src.utils.prune import Pruner
from src.utils.snip import snip
from src.utils.utils import (LrScheduler, get_lr, get_model, load_model,
                             mask_check, mask_sparsity, save_model)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = get_logger(__name__)

cifar_10_classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@register
class SparseTrainer:
    def __init__(self, args):
        self.args = args
        self.model, self.criterion, self.optimizer = get_model(self.args)
        self.best_acc = 0
        self.epoch = 0
        if args.resume:
            self.model, self.optimizer, self.epoch = load_model(
                self.model, self.optimizer, self.args.output_dir, self.args.run_name
            )
        self.best_model = self.model

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

        acc = 100.0 * correct / total

        self.model.train()
        return loss, acc

    # Training
    def train(self, trainloader, testloader):
        # Prune the model if snip
        if self.args.snip:
            self.model = snip(self.model, self.criterion, trainloader, self.args)
            self.args.end_step = 0

        self.model.train()
        pruner = Pruner(self.args, self.model)
        scheduler = LrScheduler(self.args, self.optimizer)

        logger.info("Mask check before training")
        mask_check(self.model)

        step = self.epoch * len(trainloader)
        cumulative_training_time = 0
        cumulative_train_test_time = 0

        for epoch in range(self.epoch, self.args.epochs):
            train_loss = 0
            correct = 0
            total = 0
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

            logger.info("Sparsities {}".format(mask_sparsity(self.model)))
            logger.info("-------------------")

            start_time = time.time()
            for batch_idx, data in enumerate(trainloader, 0):
                step += 1
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()

                self.optimizer.step()
                self.optimizer = scheduler.step(self.optimizer, step)
                self.model = pruner.step(self.model, step)
                train_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_time = time.time() - start_time
            end_train_time = time.time()
            if device == "cuda":
                max_memory_allocated_training = round(
                    torch.cuda.max_memory_allocated() / 10 ** 6, 2
                )
                max_memory_cached_training = round(
                    torch.cuda.max_memory_reserved() / 10 ** 6, 2
                )

                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

            # TESTING
            test_loss, test_acc = self.test(testloader)
            test_time = time.time() - end_train_time
            total_epoch_time = time.time() - start_time
            if device == "cuda":
                max_memory_allocated_testing = round(
                    torch.cuda.max_memory_allocated() / 10 ** 6, 2
                )
                max_memory_cached_testing = round(
                    torch.cuda.max_memory_reserved() / 10 ** 6, 2
                )

            cumulative_training_time += train_time
            cumulative_train_test_time += total_epoch_time

            string = (
                "Total elapsed time {:.4f} | Epoch {} | Train Loss: {:.4f} | "
                "Train Accuracy: {:.4f} | Learning Rate {:.4f} ".format(
                    cumulative_train_test_time,
                    epoch,
                    train_loss,
                    (correct / total),
                    get_lr(self.optimizer),
                )
            )
            logger.info(string)
            logger.info(
                "Train epoch time: {:.4f} | "
                "Max memory allocated training {:.4f} | "
                "Max memory cached training {:.4f}".format(
                    train_time,
                    max_memory_allocated_training,
                    max_memory_cached_training,
                )
            )
            logger.info(
                "Test epoch time: {:.4f} | "
                "Max memory allocated testing {:.4f} | "
                "Max memory cached testing {:.4f}".format(
                    test_time, max_memory_allocated_testing, max_memory_cached_testing
                )
            )
            if self.best_acc < test_acc:
                self.best_acc = test_acc
                # self.best_model = copy.deepcopy(self.model)
                save_model(
                    epoch,
                    self.model,
                    self.optimizer,
                    self.args.output_dir,
                    self.args.run_name,
                )
                logger.info("Best accuracy updated to {:.2f}".format(self.best_acc))

            # save_model(epoch, self.model, self.optimizer, self.args.output_dir,
            #           self.args.run_name)
            logger.info(
                "Test Loss: {:.4f} Test Accuracy: {:.4f}".format(test_loss, test_acc)
            )
            logger.info("-------------------")
            self.epoch = epoch

        logger.info("Training completed")
        logger.info("Best test accuracy {:.4f}".format(self.best_acc))
        # save_model(epoch, self.best_model, self.optimizer, self.args.output_dir,
        #           self.args.run_name)

        return self.best_model
