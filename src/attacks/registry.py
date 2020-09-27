_ATTACKS = dict()


def register(fn):
  global _ATTACKS
  _ATTACKS[fn.__name__] = fn
  return fn


def get_attack(model, criterion, attack_params):
  return _ATTACKS[attack_params.name](model, criterion,
                                      attack_params)
