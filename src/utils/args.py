import argparse

parser = argparse.ArgumentParser(description='Ensemble Training')
# Model Parameters
parser.add_argument(
  '--model', default='vgg19_bn', type=str, help='Model to use')
parser.add_argument(
  '--model_type', default='dense', type=str, help='Dense/sparse')
parser.add_argument(
  '--seed', default=1, type=int, help='random seed')
parser.add_argument(
  '--model', default='train', type=bool, help='In eval mode or not')
parser.add_argument(
  '--load_prev', default=False, type=bool, help='Load prev weights')

# Training params
parser.add_argument(
  '--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument(
  '--num_classes', default=10, type=int, help='Number of classes')
parser.add_argument(
  '--optim', default='sgd', type=str, help='Optimizer to use')
parser.add_argument(
  '--lr', default=0.1, type=float, help='Initial LR')
parser.add_argument(
  '--lr_schedule', default='step', type=str, help='LR scheduler')
parser.add_argument(
  '--up_step', default=10000, type=int, help='No of cycles in cyclical lr')
parser.add_argument(
  '--down_step', default=10000, type=int, help='No of cycles in cyclical lr')
parser.add_argument(
  '--milestones', default=[25000, 50000, 75000, 90000], type=list, help='Multi step lr')

parser.add_argument(
  '--epochs', default=200, type=int, help='No of epochs')
parser.add_argument(
  '--clip', default=1, type=int, help='Gradient clipping')
parser.add_argument(
  '--steps', default=None, type=int, help='No of steps')
parser.add_argument(
  '--eval_step', default=1000, type=int, help='Eval every this steps')
parser.add_argument(
  '--batch_size', default=100, type=int, help='Batch size')
parser.add_argument(
  '--output_dir',
  type=str,
  help='Output directory for storing ckpts. Default is in runs/hparams')
parser.add_argument(
  '--use_colab', type=bool, default=False, help='Use Google colaboratory')

# sparsity params
parser.add_argument(
  '--ramping', type=bool, default=False, help='Use ramping sparsity')
parser.add_argument(
  '--carry_mask', type=bool, default=False, help='Carry mask in ramping pruning')
parser.add_argument(
  '--initial_sparsity', default=0.0, type=float, help='Initial sparsity')
parser.add_argument(
  '--final_sparsity', default=0.75, type=float, help='Final sparsity')
parser.add_argument(
  '--start_step', default=0, type=float, help='Pruning start step')
parser.add_argument(
  '--end_step', default=0.2, type=float, help='Stop pruning at this step')
parser.add_argument(
  '--prune_freq', default=100, type=int, help='Prune every x steps')
parser.add_argument(
  '--global_prune', default=False, type=bool, help='Layer wise pruning')
parser.add_argument(
  '--prune_type', default='weight', type=str, help='Weight or unit pruning.')
parser.add_argument(
  '--ramp_type', default='linear', type=str, help='Ramp type.')
parser.add_argument(
  '--ramp_cycle_type', default='full', type=str, help='Ramp cycle type')


def get_args():
  args = parser.parse_args()
  if args.model_type == 'dense':
    args.output_dir = '{}/{}/lr_{}/'.format(
      args.dataset,
      args.model,
      args.lr_schedule
    )
  else:
    args.output_dir = '{}/{}/lr_{}/{}/{}_{}_{}/{}/'.format(
      args.dataset,
      args.model,
      args.lr_schedule,
      args.final_sparsity,
      args.start_step,
      args.end_step,
      args.prune_freq,
      args.ramp_type
    )

  return args
