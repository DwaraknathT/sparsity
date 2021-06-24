"""
This script handles the training process.
"""

import argparse
import math
import os
import random
import time

import dill as pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import BucketIterator, Dataset, Field
from torchtext.datasets import TranslationDataset
from tqdm import tqdm

import src.models.transformers.Constants as Constants
from src.models.transformers.Optim import ScheduledOptim
from src.models.transformers.transformer import Transformer
from src.trainers.registry import register
from src.utils.args import get_args

device = "cuda" if torch.cuda.is_available() else "cpu"


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    """ Apply label smoothing if needed """

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction="sum")
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


@register
class TransformerTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.opt.d_word_vec = opt.d_model

        # https://pytorch.org/docs/stable/notes/randomness.html
        # For reproducibility
        if self.opt.seed is not None:
            torch.manual_seed(self.opt.seed)
            torch.backends.cudnn.benchmark = False
            # torch.set_deterministic(True)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if not self.opt.output_dir:
            print("No experiment result will be saved.")

        if not os.path.exists(self.opt.output_dir):
            os.makedirs(self.opt.output_dir)

        if self.opt.batch_size < 2048 and self.opt.n_warmup_steps <= 4000:
            print(
                "[Warning] The warmup steps may be not enough.\n"
                "(sz_b, warmup) = (2048, 4000) is the official setting.\n"
                "Using smaller batch w/o longer warmup may cause "
                "the warmup stage ends with only little data trained."
            )
        # ========= Loading Dataset =========#
        if all((self.opt.train_path, self.opt.val_path)):
            training_data, validation_data = self.prepare_dataloaders_from_bpe_files(
                self.opt, device
            )
        elif self.opt.data_pkl:
            self.training_data, self.validation_data = self.prepare_dataloaders()

        print(self.opt)

        self.model = Transformer(
            self.opt.src_vocab_size,
            self.opt.trg_vocab_size,
            src_pad_idx=self.opt.src_pad_idx,
            trg_pad_idx=self.opt.trg_pad_idx,
            trg_emb_prj_weight_sharing=self.opt.proj_share_weight,
            emb_src_trg_weight_sharing=self.opt.embs_share_weight,
            d_k=self.opt.d_k,
            d_v=self.opt.d_v,
            d_model=self.opt.d_model,
            d_word_vec=self.opt.d_word_vec,
            d_inner=self.opt.d_inner_hid,
            n_layers=self.opt.n_layers,
            n_head=self.opt.n_head,
            dropout=self.opt.dropout,
            scale_emb_or_prj=self.opt.scale_emb_or_prj,
        ).to(device)

        self.optimizer = Scheduledoptim(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            self.opt.lr_mul,
            self.opt.d_model,
            self.opt.n_warmup_steps,
        )
        self.log_train_file = os.path.join(self.opt.output_dir, "train.log")
        self.log_valid_file = os.path.join(self.opt.output_dir, "valid.log")

    def prepare_dataloaders_from_bpe_files(self):
        batch_size = self.opt.batch_size
        MIN_FREQ = 2

        data = pickle.load(open(self.opt.data_pkl, "rb"))
        MAX_LEN = data["settings"].max_len
        field = data["vocab"]
        fields = (field, field)

        def filter_examples_with_length(x):
            return len(vars(x)["src"]) <= MAX_LEN and len(vars(x)["trg"]) <= MAX_LEN

        train = TranslationDataset(
            fields=fields,
            path=self.opt.train_path,
            exts=(".src", ".trg"),
            filter_pred=filter_examples_with_length,
        )
        val = TranslationDataset(
            fields=fields,
            path=self.opt.val_path,
            exts=(".src", ".trg"),
            filter_pred=filter_examples_with_length,
        )

        self.opt.max_token_seq_len = MAX_LEN + 2
        self.opt.src_pad_idx = self.opt.trg_pad_idx = field.vocab.stoi[
            Constants.PAD_WORD
        ]
        self.opt.src_vocab_size = self.opt.trg_vocab_size = len(field.vocab)

        train_iterator = BucketIterator(
            train, batch_size=batch_size, device=device, train=True
        )
        val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
        return train_iterator, val_iterator

    def prepare_dataloaders(self):
        batch_size = self.opt.batch_size
        data = pickle.load(open(self.opt.data_pkl, "rb"))

        self.opt.max_token_seq_len = data["settings"].max_len
        self.opt.src_pad_idx = data["vocab"]["src"].vocab.stoi[Constants.PAD_WORD]
        self.opt.trg_pad_idx = data["vocab"]["trg"].vocab.stoi[Constants.PAD_WORD]

        self.opt.src_vocab_size = len(data["vocab"]["src"].vocab)
        self.opt.trg_vocab_size = len(data["vocab"]["trg"].vocab)

        # ========= Preparing Model =========#
        if self.opt.embs_share_weight:
            assert (
                data["vocab"]["src"].vocab.stoi == data["vocab"]["trg"].vocab.stoi
            ), "To sharing word embedding the src/trg word2idx table shall be the same."

        fields = {"src": data["vocab"]["src"], "trg": data["vocab"]["trg"]}

        train = Dataset(examples=data["train"], fields=fields)
        val = Dataset(examples=data["valid"], fields=fields)

        train_iterator = BucketIterator(
            train, batch_size=batch_size, device=device, train=True
        )
        val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

        return train_iterator, val_iterator

    def train_epoch(self, smoothing):
        """ Epoch operation in training phase"""

        self.model.train()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        desc = "  - (Training)   "
        for batch in tqdm(self.training_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, self.opt.src_pad_idx).to(device)
            trg_seq, gold = map(
                lambda x: x.to(device), patch_trg(batch.trg, self.opt.trg_pad_idx)
            )

            # forward
            self.optimizer.zero_grad()
            pred = self.model(src_seq, trg_seq)

            # backward and update parameters
            loss, n_correct, n_word = cal_performance(
                pred, gold, self.opt.trg_pad_idx, smoothing=smoothing
            )
            loss.backward()
            self.optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

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

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return (
            loss_per_word,
            accuracy,
            max_memory_allocated_training,
            max_memory_cached_training,
        )

    def eval_epoch(self):
        """ Epoch operation in evaluation phase """

        self.model.eval()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        desc = "  - (Validation) "
        with torch.no_grad():
            for batch in tqdm(
                self.validation_data, mininterval=2, desc=desc, leave=False
            ):

                # prepare data
                src_seq = patch_src(batch.src, self.opt.src_pad_idx).to(device)
                trg_seq, gold = map(
                    lambda x: x.to(device), patch_trg(batch.trg, self.opt.trg_pad_idx)
                )

                # forward
                pred = self.model(src_seq, trg_seq)
                loss, n_correct, n_word = cal_performance(
                    pred, gold, self.opt.trg_pad_idx, smoothing=False
                )

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

        if device == "cuda":
            max_memory_allocated_testing = round(
                torch.cuda.max_memory_allocated() / 10 ** 6, 2
            )
            max_memory_cached_testing = round(
                torch.cuda.max_memory_reserved() / 10 ** 6, 2
            )

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return (
            loss_per_word,
            accuracy,
            max_memory_allocated_testing,
            max_memory_cached_testing,
        )

    def train(self):
        """ Start training """

        # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
        if self.opt.use_tb:
            print("[Info] Use Tensorboard")
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(
                log_dir=os.path.join(self.opt.output_dir, "tensorboard")
            )

        log_train_file = os.path.join(self.opt.output_dir, "train.log")
        log_valid_file = os.path.join(self.opt.output_dir, "valid.log")

        print(
            "[Info] Training performance will be written to file: {} and {}".format(
                log_train_file, log_valid_file
            )
        )

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            log_vf.write("epoch,loss,ppl,accuracy\n")

        def print_performances(header, ppl, accu, time, lr, max_mem, cache_mem):
            print(
                "  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, "
                "Max mem allocated: {max_mem} Max mem cached: {cache_mem} elapse: {elapse:3.3f} min ".format(
                    header=f"({header})",
                    ppl=ppl,
                    accu=100 * accu,
                    lr=lr,
                    max_mem=max_mem,
                    cache_mem=cache_mem,
                    elapse=(time) / 60,
                )
            )

        # valid_accus = []
        valid_losses = []
        for epoch_i in range(self.opt.epoch):
            print("[ Epoch", epoch_i, "]")

            train_start_time = time.time()
            (
                train_loss,
                train_accu,
                max_memory_allocated_training,
                max_memory_cached_training,
            ) = self.train_epoch(smoothing=self.opt.label_smoothing)
            train_end_time = time.time()
            train_ppl = math.exp(min(train_loss, 100))
            # Current learning rate
            lr = self.optimizer._self.optimizer.param_groups[0]["lr"]
            print_performances(
                "Training",
                train_ppl,
                train_accu,
                (train_end_time - train_start_time),
                lr,
                max_memory_allocated_training,
                max_memory_cached_training,
            )

            eval_start_time = time.time()
            (
                valid_loss,
                valid_accu,
                max_memory_allocated_testing,
                max_memory_cached_testing,
            ) = self.eval_epoch()
            eval_end_time = time.time()
            valid_ppl = math.exp(min(valid_loss, 100))
            print_performances(
                "Validation",
                valid_ppl,
                valid_accu,
                (eval_end_time - eval_start_time),
                lr,
                max_memory_allocated_testing,
                max_memory_cached_testing,
            )

            valid_losses += [valid_loss]

            checkpoint = {
                "epoch": epoch_i,
                "settings": self.opt,
                "model": self.model.state_dict(),
            }

            if self.opt.save_mode == "all":
                model_name = "model_accu_{accu:3.3f}.chkpt".format(
                    accu=100 * valid_accu
                )
                torch.save(checkpoint, model_name)
            elif self.opt.save_mode == "best":
                model_name = "model.chkpt"
                if valid_loss <= min(valid_losses):
                    torch.save(
                        checkpoint, os.path.join(self.opt.output_dir, model_name)
                    )
                    print("    - [Info] The checkpoint file has been updated.")

            with open(self.log_train_file, "a") as log_tf, open(
                self.log_valid_file, "a"
            ) as log_vf:
                log_tf.write(
                    "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f},{max_mem},{cache_mem},{elapse:3.3f}\n".format(
                        epoch=epoch_i,
                        loss=train_loss,
                        ppl=train_ppl,
                        accu=100 * train_accu,
                        max_mem=max_memory_allocated_training,
                        cache_mem=max_memory_cached_training,
                        elapse=(train_end_time - train_start_time),
                    )
                )
                log_vf.write(
                    "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f},{max_mem},{cache_mem},{elapse:3.3f}\n".format(
                        epoch=epoch_i,
                        loss=valid_loss,
                        ppl=valid_ppl,
                        accu=100 * valid_accu,
                        max_mem=max_memory_allocated_testing,
                        cache_mem=max_memory_cached_testing,
                        elapse=(eval_end_time - eval_start_time),
                    )
                )

            if self.opt.use_tb:
                tb_writer.add_scalars(
                    "ppl", {"train": train_ppl, "val": valid_ppl}, epoch_i
                )
                tb_writer.add_scalars(
                    "accuracy",
                    {"train": train_accu * 100, "val": valid_accu * 100},
                    epoch_i,
                )
                tb_writer.add_scalar("learning_rate", lr, epoch_i)
