from src.hparams.base import Dense, Sparse
from .registry import register


@register
def resnet32_dense():
  hps = Dense()
  hps.model = 'resnet32'
  return hps


# =======================================
# Sparse models
# elements in the function name :
# weight/unit pruning
# linear, cyclic, quadratic ramping
# final sparsity
# start step
# end step
# prune frequency
# global, local prune
# learning rate scheduler
# carry mask, no carry mask - cm, ncm
# ========================================

@register
def weight_linear_90_0_03_100_local_ncm_step():
  hps = Sparse()
  hps.model = 'resnet32'
  hps.final_sparsity = 0.9
  hps.end_step = 0.3
  hps.name = 'weight_linear_90_0_03_100_local_ncm_step'

  return hps


@register
def weight_linear_90_0_03_100_local_cm_step():
  hps = Sparse()
  hps.model = 'resnet32'
  hps.final_sparsity = 0.9
  hps.end_step = 0.3
  hps.carry_mask = True
  hps.name = 'weight_linear_90_0_03_100_local_cm_step'

  return hps


@register
def weight_linear_90_0_05_100_local_ncm_step():
  hps = Sparse()
  hps.model = 'resnet32'
  hps.final_sparsity = 0.9
  hps.name = 'weight_linear_90_0_05_100_local_ncm_step'

  return hps


@register
def weight_linear_90_0_05_100_local_cm_step():
  hps = Sparse()
  hps.model = 'resnet32'
  hps.final_sparsity = 0.9
  hps.carry_mask = True
  hps.name = 'weight_linear_90_0_05_100_local_cm_step'

  return hps
