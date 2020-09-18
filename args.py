import argparse

parser = argparse.ArgumentParser(description='Sparsity')
# Model Parameters
parser.add_argument(
  '--model', default='vgg19', type=str, help='Model to use')
parser.add_argument(
  '--model', default='vgg19', type=str, help='Model to use')
parser.add_argument(
  '--seed', default=5, type=int, help='random seed')
parser.add_argument(
  '--model_type', default='dense', type=str, help='dense or sparse')
parser.add_argument(
  '--eval', default=False, type=bool, help='In eval mode or not')

# Training params
parser.add_argument(
  '--dataset', default='cifar10', type=str, help='Dataset to use cifar10,100')
parser.add_argument(
  '--num_classes', default=10, type=int, help='Number of classes')
parser.add_argument(
  '--optim', default='sgd', type=str, help='Optimizer to use')
parser.add_argument(
  '--lr', default=0.1, type=float, help='Initial LR')
parser.add_argument(
  '--lr_schedule', default='step', type=str, help='LR scheduler')
parser.add_argument(
  '--cycle_div', default=None, type=int, help='No of cycles in cyclical lr')
parser.add_argument(
  '--up_step', default=None, type=int, help='No of cycles in cyclical lr')
parser.add_argument(
  '--down_step', default=None, type=int, help='No of cycles in cyclical lr')


def get_args():
  args = parser.parse_args()

  return args
