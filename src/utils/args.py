import argparse

from src.hparams.registry import get_hparams

parser = argparse.ArgumentParser()
parser.add_argument(
  '--hparams', type=str, required=True, help='Hyperparameters string')
parser.add_argument(
  '--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument(
  '--seed', default=1, type=int, help='Seed for random inits')
parser.add_argument(
  '--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument(
  '--use_colab', type=bool, default=False, help='Use Google colaboratory')
args = parser.parse_args()


def get_args():
  args = parser.parse_args()
  hparams = get_hparams(args.hparams)
  hparams.use_colab = args.use_colab
  hparams.resume = args.resume
  hparams.dataset = args.dataset

  return hparams
