"""User-reviewed promotion of Zoller's West Pahari varieties to language records.

Keep this separate from extraction so source spans and persistent keys do not change.
The source's broad Western label remains explicitly Western West Pahari; no
unattributed form is assigned to a more specific language.
"""
import json
from pathlib import Path

PROMOTIONS = json.loads(Path(__file__).with_name('west-pahari-languages.json').read_text())


def resolve_language(language, dialect):
    record = PROMOTIONS.get(dialect)
    if record and (language in ('WPah', record['ID']) or
                   (language == 'Garh' and dialect == 'Bangani')):
        return record['ID'], ''
    return language, dialect


def promote_mapping(mapping):
    for item in mapping.values():
        old = item['language'], item['dialect']
        new = resolve_language(*old)
        if new != old:
            item['language'], item['dialect'] = new
            item['basis'] = 'Explicit editorial decision 2026-09-13: separate language in W. Pahari clade; source label preserved'
