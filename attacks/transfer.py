"""Craft PGD examples using a separately supplied surrogate, without target queries."""

from attacks.white_box import pgd


def transfer_pgd(surrogate, original, label, **kwargs):
    result = pgd(surrogate, original, label, **kwargs)
    result.details["surrogate_calls"] = result.calls
    result.details["target_attack_queries"] = 0
    return result
