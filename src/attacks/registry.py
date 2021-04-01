_ATTACKS = dict()


def register(fn):
    global _ATTACKS
    _ATTACKS[fn.__name__] = fn
    return fn


def get_attack(criterion, attack_params):
    return _ATTACKS[attack_params.name](criterion, attack_params)
