import argparse
from src.hparams.registry import get_hparams
from src.utils.datasets import get_data
from src.utils.logger import get_logger, setup_dirs

parser = argparse.ArgumentParser()
parser.add_argument(
  '--hparams', type=str, required=True, help='Hyperparameters string')
parser.add_argument(
  '--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument(
  '--use_colab', type=bool, default=False, help='Use Google colaboratory')
args = parser.parse_args()

hparams = get_hparams(args.hparams)
#hparams.use_colab = args.use_colab
#setup_dirs(hparams)
print(hparams)

