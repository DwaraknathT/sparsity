from src.hparams.base import Dense, Sparse
from .registry import register


@register
def vgg19_dense():
  hps = Dense()
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
def weight_linear_50_0_05_100_local_ncm_step():
  hps = Sparse()
  hps.final_sparsity = 0.5
  hps.name = 'weight_linear_50_0_05_100_local_ncm_step'

  return hps


@register
def weight_linear_50_0_05_100_local_cm_step():
  hps = Sparse()
  hps.final_sparsity = 0.5
  hps.carry_mask = True


  return hps


@register
def weight_linear_50_0_05_100_global_cm_step():
  hps = Sparse()
  hps.final_sparsity = 0.5
  hps.global_prune = True
  hps.carry_mask = True

  return hps


@register
def weight_linear_50_0_05_100_global_ncm_step():
  hps = Sparse()
  hps.final_sparsity = 0.5
  hps.global_prune = True

  return hps


# ==============================================
# Targeted Unit
# ==============================================
