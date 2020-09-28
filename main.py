import numpy as np
import torch

from src.trainers.dense_trainer import DenseTrainer
from src.trainers.sparse_trainer import SparseTrainer
from src.utils.args import get_args
from src.utils.datasets import get_data
from src.utils.logger import get_logger

# Get parameters and setup directories, loggers
args = get_args()
logger = get_logger(__name__)

# Seed seeds for reproducibility
np.random.seed(args.seed)  # cpu vars
torch.manual_seed(args.seed)  # cpu  vars
torch.cuda.manual_seed_all(args.seed)  # gpu vars

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device

trainers = {
  'dense': DenseTrainer,
  'sparse': SparseTrainer
}


def main():
  trainloader, testloader = get_data(args)
  # get trainer
  trainer = trainers[args.model_type](args)
  model = trainer.train(trainloader, testloader)
  trainer.test_attack(args.attack, testloader)


if __name__ == '__main__':
  main()
