class Dense:
  def __init__(self):
    # Model params
    self.model = 'vgg11_bn'
    self.model_type = 'dense'
    self.seed = 1
    self.mode = 'train'
    self.device = 'cpu'

    # Training params
    self.epochs = 200
    self.steps = None
    self.eval_step = 1000
    self.clip = 1
    self.dataset = 'cifar10'
    self.num_classes = 10
    self.batch_size = 100
    self.optim = 'sgd'
    self.lr = 0.1
    self.lr_schedule = 'step'
    self.milestones = [25000, 50000, 75000, 90000]

    # Cyclic lr params
    self.up_step = 10000
    self.down_step = 10000

    # Output name
    self.output_dir = '{}/{}/lr_{}/'.format(
      self.dataset,
      self.model,
      self.lr_schedule
    )


class Sparse:
  def __init__(self):
    # Model params
    self.model = 'vgg19_bn'
    self.model_type = 'sparse'
    self.seed = 1
    self.mode = 'train'
    self.device = 'cpu'

    # Training params
    self.epochs = 200
    self.steps = None
    self.eval_step = 1000
    self.clip = 1
    self.dataset = 'cifar10'
    self.num_classes = 10
    self.batch_size = 100
    self.optim = 'sgd'
    self.lr = 0.1
    self.lr_schedule = 'step'
    self.milestones = [25000, 50000, 75000, 90000]

    # Cyclic lr params
    self.up_step = 10000
    self.down_step = 10000

    self.model_type = 'sparse'
    self.ramping = False
    self.carry_mask = False

    # Sparsity params
    self.initial_sparsity = 0
    self.final_sparsity = 0.75
    self.start_step = 0
    self.end_step = 0.5
    self.prune_freq = 100
    self.global_prune = False  # global, local prning
    self.prune_type = 'weight'  # weight, unit pruning
    self.ramp_type = 'linear'  # linear, cyclic, quadratic ramp

    # Output name 
    self.output_dir = '{}/{}/lr_{}/{}/{}_{}_{}/{}/'.format(
      self.dataset,
      self.model,
      self.lr_schedule,
      self.final_sparsity,
      self.start_step,
      self.end_step,
      self.prune_freq,
      self.ramp_type
    )
