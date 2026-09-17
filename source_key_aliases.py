"""Preserve public IDs when a reviewed source correction rejoins parse fragments."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def apply_source_key_aliases(aliases, previous_registry, next_registry, active_ids, paths=None):
    paths = paths if paths is not None else sorted((ROOT / 'data/other/form_aliases').glob('*.csv'))
    active_keys = {r['Source_Key']: r['Form_ID'] for r in next_registry
                   if r.get('Status') == 'active' and r.get('Source_Key')}
    previous = defaultdict(set)
    for row in previous_registry:
        if row.get('Source_Key'):
            previous[row['Source_Key']].add(row['Form_ID'])
    replacements = {}
    for path in paths:
        with Path(path).open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                old, target = row['Retired_Source_Key'], row['Target_Source_Key']
                if old == target or not row['Reason'].strip():
                    raise ValueError(f'{path}: invalid source alias {old}')
                if target not in active_keys:
                    raise ValueError(f'{path}: missing active source alias target {target}')
                if old in active_keys:
                    raise ValueError(f'{path}: cannot alias still-active source key {old}')
                for old_id in previous[old]:
                    if old_id == active_keys[target]:
                        continue
                    if old_id in active_ids and old_id != active_keys[target]:
                        raise ValueError(f'{path}: source alias would replace active ID {old_id}')
                    if old_id in replacements and replacements[old_id] != active_keys[target]:
                        raise ValueError(f'{path}: conflicting source alias for {old_id}')
                    replacements[old_id] = active_keys[target]
    # Existing legacy URLs that ended at a retired opaque ID must follow it too.
    for alias, target in list(aliases.items()):
        if target in replacements:
            aliases[alias] = replacements[target]
    aliases.update(replacements)
    return len(replacements)
