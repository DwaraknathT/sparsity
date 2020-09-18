from src.hparams.base import Dense
from .registry import register


@register
def resnet32_dense():
  hps = Dense()
  return hps

