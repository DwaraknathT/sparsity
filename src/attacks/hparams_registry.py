_APARAMS = dict()


def register(fn):
  global _HPARAMS
  _APARAMS[fn.__name__] = fn()
  return fn


def get_attack_params(name=None):
  if name is None:
    return _APARAMS
  return _APARAMS[name]
