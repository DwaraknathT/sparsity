from src.attacks.hparams.registry import register


class HParams():

  def __init__(self):
    self.targeted = False
    self.eval_steps = 100
    self.min_value = 0
    self.max_value = 1
    self.batch_size = 100
    self.epsilons = None


@register
def fgsm():
  hps = HParams()
  hps.name = "FGSM"
  hps.epsilons = [x / 255 for x in range(0, 4)]
  return hps


@register
def pgd():
  hps = HParams()
  hps.name = "PGD"
  hps.eval_steps = 100
  hps.random_start = True
  hps.iters = 40
  hps.step_size = 0.01
  hps.epsilons = [x / 255 for x in range(0, 9)]
  return hps
