import numpy as np
import torch

from src.models.registry import get_model
from src.utils.args import get_args
from src.utils.datasets import get_data
from src.utils.logger import get_logger, setup_dirs
from src.trainers.dense_trainer import train, test

# Get paramodelmeters and setup directories, loggers
args = get_args()
setup_dirs()
logger = get_logger(__name__)

# Seed seeds for reproducibility
np.random.seed(args.seed)  # cpu vars
torch.manual_seed(args.seed)  # cpu  vars
torch.cuda.manual_seed_all(args.seed)  # gpu vars

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device


def main():
  trainloader, testloader = get_data(args)
  train(trainloader, testloader)


if __name__ == '__main__':
  main()
