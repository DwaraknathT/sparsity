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
  hps.eps = [x / 255 for x in range(0, 9)]
  return hps


@register
def pgd():
  hps = HParams()
  hps.name = "PGD"
  hps.random_start = True
  hps.steps = 40
  hps.alpha = 2 / 255
  hps.eps = [x / 255 for x in range(0, 9)]
  return hps


@register
def apgd():
  hps = HParams()
  hps.name = "APGD"
  hps.random_start = True
  hps.steps = 40
  hps.alpha = 2 / 255
  hps.eps = [x / 255 for x in range(0, 9)]
  return hps
