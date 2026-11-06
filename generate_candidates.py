"""Rule evaluators and compatibility entry point for predict_top15.py."""
from __future__ import annotations

import numpy as np

from combinatorial import evaluate_spec


def evaluate_next(x, features, spec):
    if spec.get('kind') == 'pool':
        count = np.isin(x, spec['pool']).sum(axis=1)
        return count == spec['value'] if spec['op'] == 'eq' else count >= spec['value']
    if spec.get('kind', '').startswith('bonus'):
        raise ValueError('Bonus-only rules do not constrain a six-number ticket')
    return evaluate_spec(features, spec)


def compatible_bonus(x, specs):
    numbers = np.arange(1, 46)
    allowed = ~np.isin(numbers, x)
    for spec in specs:
        if spec['kind'] == 'bonus_sum_mod':
            allowed &= np.isin((numbers + sum(x)) % spec['mod'], spec['residues'])
        elif spec['feature'] == 'bonus_even': allowed &= numbers % 2 == 0
        elif spec['feature'] == 'bonus_low': allowed &= numbers <= 22
        elif spec['feature'] == 'bonus_below_x3': allowed &= numbers < x[2]
        elif spec['feature'] == 'bonus_above_x4': allowed &= numbers > x[3]
        else: raise ValueError(spec)
    return numbers[allowed]


def main():
    from predict_top15 import main as predict_main
    return predict_main()


if __name__ == '__main__':
    main()
