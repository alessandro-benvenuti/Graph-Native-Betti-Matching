"""Experimental compatibility for historical resolved checkpoint configs.

Only later topology schema fields are backfilled. Supplied values and all
classification/model/data/loss coefficients stay intact; additions are audited.
"""
import copy
import os
from pathlib import Path

import yaml
from configs import ConfigError, validate_config
from configs.loader import _expand_environment, _load_with_defaults

ROOT = Path(__file__).resolve().parents[2]


def load_source_config(path, *, environment=None):
    original = _expand_environment(_load_with_defaults(Path(path), set()),
        os.environ if environment is None else environment, 'config')
    config = copy.deepcopy(original)
    topology = config.get('topology')
    if not isinstance(topology, dict):
        raise ConfigError('Historical source topology must be a mapping')
    base_path = ROOT / 'configs/base.yaml'
    base = yaml.safe_load(base_path.read_text())['topology']
    additions = []

    def fill_missing(target, defaults, prefix):
        for key, value in defaults.items():
            name = prefix + '.' + key
            if key not in target or (isinstance(value,dict) and target[key] is None):
                reason = 'historical null block' if key in target else 'absent in historical config'
                target[key] = copy.deepcopy(value)
                if isinstance(value, dict):
                    record_leaves(value, name, reason)
                else:
                    additions.append(dict(path=name, value=value, reason=reason))
            elif isinstance(value, dict) and isinstance(target[key], dict):
                fill_missing(target[key], value, name)

    def record_leaves(value, prefix, reason):
        for key, item in value.items():
            name = prefix + '.' + key
            if isinstance(item, dict):
                record_leaves(item, name, reason)
            else:
                additions.append(dict(path=name, value=item, reason=reason))

    # These blocks were introduced after the epoch-500 focal checkpoint.
    # Their current checked-in defaults are explicit provenance, not a rewrite
    # of settings supplied by the historical run.
    defaults = {key:base[key] for key in ('sampling', 'complex')}
    for key in ('betti_h0', 'betti_h1'):
        defaults[key] = {name:base[key][name] for name in ('normalization', 'activation_epoch')}
    fill_missing(topology, defaults, 'topology')
    validate_config(config)
    audit = dict(policy='missing topology schema fields only; supplied values preserved',
        defaults_file='configs/base.yaml', additions=additions)
    return config, original, audit
