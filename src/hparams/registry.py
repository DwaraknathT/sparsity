_HPARAMS = dict()


def register(fn):
  global _HPARAMS
  print(fn.__name__)
  _HPARAMS[fn.__name__] = fn()
  return fn


def get_hparams(name=None):
  if name is None:
    return _HPARAMS
  return _HPARAMS[name]
